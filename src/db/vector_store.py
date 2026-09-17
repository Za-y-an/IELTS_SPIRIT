from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import Settings


from pathlib import Path

def get_qdrant_client(settings: Settings, force_memory: bool = False) -> QdrantClient:
    """
    Factory function for QdrantClient.
    Explicitly branches on whether QDRANT_URL is provided:
    - If QDRANT_URL is provided: connects to the remote cluster via URL and optional API key.
    - If QDRANT_URL is empty / unset: uses local storage (data/qdrant_storage) so vectors persist
      across CLI executions, or in-memory mode if force_memory=True.
    
    NEVER conflates ':memory:' with a URL string.
    """
    if settings.is_qdrant_in_memory:
        if force_memory:
            return QdrantClient(location=":memory:")
        storage_path = Path(__file__).resolve().parent.parent.parent / "data" / "qdrant_storage"
        storage_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(storage_path))

    return QdrantClient(
        url=settings.qdrant_url.strip(),
        api_key=settings.qdrant_api_key.strip() if settings.qdrant_api_key else None,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 3072,
) -> bool:
    """
    Ensures that the specified Qdrant collection exists and matches the target vector dimension.
    If a stale collection already exists with the wrong dimension (e.g. legacy 1536d vs 3072d),
    drops and recreates it with Cosine distance metric.
    Returns True if collection exists or was created successfully.
    """
    if client.collection_exists(collection_name=collection_name):
        # Inspect existing vector dimension
        try:
            col_info = client.get_collection(collection_name=collection_name)
            vectors_config = col_info.config.params.vectors
            existing_size = None
            if hasattr(vectors_config, "size"):
                existing_size = vectors_config.size
            elif isinstance(vectors_config, dict) and "" in vectors_config:
                existing_size = getattr(vectors_config[""], "size", None)

            if existing_size is not None and existing_size != vector_size:
                # Dimension mismatch: drop stale collection and recreate
                client.delete_collection(collection_name=collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                return True
        except Exception:
            # If inspection fails for any reason, proceed
            pass
        return True

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    return True
