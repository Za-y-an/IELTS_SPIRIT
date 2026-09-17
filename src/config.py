from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    
    NOTE ON SUPABASE POSTGRESQL CONNECTION STRING:
    Supabase's dashboard provides a standard sync connection string:
      postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
    or for the transaction pooler (port 6543):
      postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
      
    To use with async SQLAlchemy + asyncpg, this string must be transformed:
      1. Driver prefix: Change 'postgresql://' to 'postgresql+asyncpg://'
      2. Special characters in password: Must be percent-encoded (e.g. '@' -> '%40', '#' -> '%23', '=' -> '%3D')
      3. SSL requirement: Supabase requires SSL encryption. Append '?ssl=require' or pass
         connect_args={"ssl": "require"} to create_async_engine.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini configuration
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=3072, alias="EMBEDDING_DIMENSION")
    generation_model: str = Field(default="gemini-3.6-flash", alias="GENERATION_MODEL")

    # Qdrant configuration (blank QDRANT_URL signifies in-memory mode)
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="ielts_knowledge", alias="QDRANT_COLLECTION_NAME")

    # Supabase PostgreSQL (asyncpg format: postgresql+asyncpg://user:pass@host:port/dbname?ssl=require)
    supabase_db_url: str = Field(default="", alias="SUPABASE_DB_URL")

    # Ingestion chunking defaults
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    @property
    def is_qdrant_in_memory(self) -> bool:
        """Returns True if Qdrant should run in local in-memory mode."""
        return not bool(self.qdrant_url and self.qdrant_url.strip())

    def masked_summary(self) -> dict[str, str]:
        """Produce a dictionary of configuration values with sensitive tokens masked."""
        def _mask(secret: Optional[str], show_prefix: int = 4, show_suffix: int = 4) -> str:
            if not secret or not secret.strip():
                return "[dim]Not Set / None[/dim]"
            s = secret.strip()
            if len(s) <= (show_prefix + show_suffix + 2):
                return "********"
            return f"{s[:show_prefix]}...{s[-show_suffix:]}"

        return {
            "Gemini API Key": _mask(self.gemini_api_key, show_prefix=3, show_suffix=4),
            "Embedding Model": f"{self.embedding_model} ({self.embedding_dimension}d)",
            "Qdrant Target": "[bold cyan]:memory: (Local In-Memory)[/bold cyan]" if self.is_qdrant_in_memory else f"Remote ({self.qdrant_url.strip()})",
            "Qdrant API Key": "[dim]N/A (in-memory)[/dim]" if self.is_qdrant_in_memory else _mask(self.qdrant_api_key),
            "Qdrant Collection": self.qdrant_collection_name,
            "Supabase DB URL": _mask(self.supabase_db_url, show_prefix=22, show_suffix=12),
            "Chunk Configuration": f"size={self.chunk_size} chars, overlap={self.chunk_overlap} chars",
        }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
