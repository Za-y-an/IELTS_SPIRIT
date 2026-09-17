from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from config import Settings, get_settings
from db.vector_store import get_qdrant_client
from ingestion.embedder import GeminiEmbedder, query_qdrant_similar


SYSTEM_INSTRUCTION = """You are a precise, grounded AI assistant for IELTS Spirit.

CRITICAL GUARDRAILS:
1. You must answer the user's question STRICTLY and ONLY based on the facts directly stated in the provided Context below.
2. Do NOT use outside knowledge, extrapolate beyond the text, or guess.
3. If the provided Context does not contain sufficient facts to answer the question, state clearly:
   "I do not have enough information in the provided document to answer this question."
4. Be factual, clear, and direct. Do not mention that you were given a prompt or snippets unless relevant to formatting.
"""


async def answer_question(
    query: str,
    client: Optional[QdrantClient] = None,
    settings: Optional[Settings] = None,
    collection_name: Optional[str] = None,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    RAG query engine pipeline:
    1. Embed query using gemini-embedding-001 (dimension 3072, exactly matching ingestion).
    2. Retrieve top-k chunks from Qdrant collection.
    3. Filter by similarity threshold (default 0.5). If no chunk passes, return early without generating.
    4. Construct anti-hallucination prompt with retrieved context.
    5. Call Gemini chat model (gemini-3.6-flash) to generate a grounded response.
    6. Return dictionary containing answer, retrieved chunks, scores, and status metadata.
    """
    cfg = settings or get_settings()
    col_name = collection_name or cfg.qdrant_collection_name

    if not cfg.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured in settings or environment.")

    # 1. Initialize vector store client if not provided
    qdrant = client or get_qdrant_client(cfg)

    # 2. Check if collection exists and has points before calling embedding API
    if not qdrant.collection_exists(col_name):
        return {
            "answer": f"Collection '{col_name}' does not exist in the vector store. Please ingest documents first.",
            "chunks": [],
            "scores": [],
            "retrieval_status": "collection_not_found",
            "model_used": None,
        }

    col_info = qdrant.get_collection(col_name)
    if col_info.points_count == 0:
        return {
            "answer": "No indexed content was found in the vector store.",
            "chunks": [],
            "scores": [],
            "retrieval_status": "empty_collection",
            "model_used": None,
        }

    # 3. Embed the query using the exact model/dimension used during ingestion
    embedder = GeminiEmbedder(
        api_key=cfg.gemini_api_key,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
    )
    query_vector = await embedder.get_query_embedding(query)

    search_hits = query_qdrant_similar(
        client=qdrant,
        collection_name=col_name,
        query_vector=query_vector,
        limit=top_k,
    )

    if not search_hits:
        return {
            "answer": "No indexed content was found in the vector store.",
            "chunks": [],
            "scores": [],
            "retrieval_status": "empty_collection",
            "model_used": None,
        }

    all_scores = [hit.score for hit in search_hits]
    all_chunks = [hit.payload for hit in search_hits if hit.payload]

    # 4. Check similarity threshold guardrail
    valid_hits = [hit for hit in search_hits if hit.score >= similarity_threshold]
    if not valid_hits:
        return {
            "answer": "I do not have enough relevant information in the provided document to answer your question.",
            "chunks": all_chunks,
            "scores": all_scores,
            "retrieval_status": "insufficient_similarity",
            "model_used": None,
        }

    # 5. Build grounded context for Gemini
    context_sections: List[str] = []
    for idx, hit in enumerate(valid_hits, start=1):
        payload = hit.payload or {}
        snippet = payload.get("text", "").strip()
        page_num = payload.get("page", "unknown")
        context_sections.append(f"[Snippet {idx} - Page {page_num}]:\n{snippet}")

    context_str = "\n\n".join(context_sections)

    user_prompt = f"""--- CONTEXT BEGIN ---
{context_str}
--- CONTEXT END ---

User Question: {query}

Answer based ONLY on the context above:"""

    # 6. Call Gemini generation model with resilient 503 fallback
    gen_client = genai.Client(api_key=cfg.gemini_api_key)
    candidate_models = [cfg.generation_model]
    if "gemini-3.5-flash-lite" not in candidate_models:
        candidate_models.append("gemini-3.5-flash-lite")

    answer_text = ""
    used_model = cfg.generation_model
    last_err: Optional[Exception] = None

    for model_name in candidate_models:
        try:
            chat = gen_client.aio.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                ),
            )
            response = await chat.send_message(user_prompt)
            answer_text = response.text.strip() if response.text else "No response generated."
            used_model = model_name
            break
        except Exception as exc:
            last_err = exc
            continue

    if not answer_text and last_err:
        raise last_err

    return {
        "answer": answer_text,
        "chunks": [hit.payload for hit in valid_hits if hit.payload],
        "scores": [hit.score for hit in valid_hits],
        "retrieval_status": "success",
        "model_used": used_model,
    }
