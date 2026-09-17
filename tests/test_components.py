import pytest
import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Settings
from db.vector_store import get_qdrant_client, ensure_collection
from ingestion.chunker import RecursiveTextSplitter
from ingestion.embedder import upsert_chunks_to_qdrant, query_qdrant_similar, generate_deterministic_mock_vector


def test_settings_masking_and_in_memory():
    s = Settings(
        GEMINI_API_KEY="AIzaSy1234567890abcdef1234567890",
        QDRANT_URL="",
        SUPABASE_DB_URL="postgresql+asyncpg://postgres:secret123@db.supabase.co:5432/postgres?ssl=require",
    )
    assert s.is_qdrant_in_memory is True
    summary = s.masked_summary()
    assert "AIz...7890" in summary["Gemini API Key"]
    assert ":memory:" in summary["Qdrant Target"]
    assert "secret123" not in summary["Supabase DB URL"]
    assert "3072d" in summary["Embedding Model"]


def test_recursive_chunker_splitting():
    text = (
        "IELTS Academic Reading test consists of three sections.\n\n"
        "Section 1 contains two or three factual texts about everyday subjects.\n\n"
        "Section 2 focuses on work-related issues, such as applying for a job, company policies, and workplace facilities.\n\n"
        "Section 3 contains one long text on a topic of general interest."
    )
    splitter = RecursiveTextSplitter(chunk_size=120, chunk_overlap=30)
    pages = [{"page": 1, "text": text, "source": "reading_guide.pdf"}]
    chunks = splitter.split_pages(pages)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.char_count <= 130
        assert chunk.source == "reading_guide.pdf"
        assert chunk.page == 1


def test_qdrant_in_memory_3072d_lifecycle():
    s = Settings(QDRANT_URL="")
    client = get_qdrant_client(s)
    collection_name = "test_collection_gemini"

    # Ensure 3072d collection
    created = ensure_collection(client, collection_name=collection_name, vector_size=3072)
    assert created is True
    assert client.collection_exists(collection_name=collection_name) is True

    # Test stale dimension drop and recreate
    # Create with 1536 first
    legacy_col = "test_legacy_stale"
    ensure_collection(client, collection_name=legacy_col, vector_size=1536)
    assert client.get_collection(legacy_col).config.params.vectors.size == 1536
    # Call ensure_collection with 3072 -> should detect mismatch, recreate with 3072
    ensure_collection(client, collection_name=legacy_col, vector_size=3072)
    assert client.get_collection(legacy_col).config.params.vectors.size == 3072

    # Upsert chunks
    splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=40)
    pages = [
        {"page": 1, "text": "IELTS Speaking Part 1 consists of familiar topics like hobbies and studies.", "source": "speaking.pdf"},
        {"page": 2, "text": "IELTS Speaking Part 2 requires speaking for two minutes on a given cue card.", "source": "speaking.pdf"},
    ]
    chunks = splitter.split_pages(pages)
    embeddings = [generate_deterministic_mock_vector(c.text, 3072) for c in chunks]

    upserted = upsert_chunks_to_qdrant(client, collection_name, chunks, embeddings)
    assert upserted == len(chunks)

    # Search
    query_vec = generate_deterministic_mock_vector("cue card speaking topic", 3072)
    results = query_qdrant_similar(client, collection_name, query_vec, limit=2)
    assert len(results) > 0
    assert "speaking.pdf" == results[0].payload["source"]


def test_rag_threshold_handling():
    import asyncio
    from rag.engine import answer_question
    s = Settings(
        GEMINI_API_KEY="AIzaSyDummyKeyForThresholdTest12345",
        QDRANT_URL="",
    )
    client = get_qdrant_client(s, force_memory=True)
    ensure_collection(client, "test_threshold_col", 3072)

    async def _run():
        return await answer_question(
            query="Unrelated question",
            client=client,
            settings=s,
            collection_name="test_threshold_col",
            similarity_threshold=0.99,
        )

    # Empty collection triggers empty_collection check before embedding call
    result = asyncio.run(_run())
    assert result["retrieval_status"] == "empty_collection"
    client.close()

