from typing import Any, List
import hashlib
import random
import uuid
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from ingestion.chunker import TextChunk


class GeminiEmbedder:
    """Handles generating embeddings via Google's Gemini SDK (gemini-embedding-001)."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dimension: int = 3072):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)

    async def get_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Fetch vector embeddings asynchronously for a list of texts using Gemini.
        Batches requests into chunks of <= 100 (default 50) and includes automatic
        exponential backoff on rate limits (HTTP 429 / RESOURCE_EXHAUSTED).
        Returns a list of float vectors (default length 3072).
        """
        if not texts:
            return []

        import asyncio
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await self.client.aio.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                    all_embeddings.extend([list(item.values) for item in response.embeddings])
                    break
                except Exception as exc:
                    err_str = str(exc)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                        wait_time = 12 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            # Brief courtesy delay between batches to respect rate quotas
            if i + batch_size < len(texts):
                await asyncio.sleep(1.0)

        return all_embeddings



    async def get_query_embedding(self, query: str) -> List[float]:
        """Fetch vector embedding for a search query string."""
        results = await self.get_embeddings([query])
        return results[0]


def generate_deterministic_mock_vector(text: str, dimension: int = 3072) -> List[float]:
    """
    Generate a deterministic normalized pseudo-vector from text.
    ONLY for use when explicit --offline or --fake-embeddings flag is supplied.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dimension)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


def upsert_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[TextChunk],
    embeddings: List[List[float]],
) -> int:
    """
    Upsert text chunks and their embeddings into the specified Qdrant collection.
    Returns the number of points inserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
        )

    points: List[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source}_{chunk.chunk_index}_{chunk.page}"))
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk.to_dict(),
            )
        )

    batch_size = 40
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
    return len(points)



def query_qdrant_similar(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 3,
) -> List[Any]:
    """
    Query the Qdrant collection for top similar chunks by vector.
    Uses modern query_points API with fallback to search.
    """
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return res.points
    elif hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
    return []
