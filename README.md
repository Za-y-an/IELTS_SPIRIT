# IELTS Spirit - WhatsApp Support Agent (RAG Ingestion & Persistence)

An async-first ingestion and persistence backend for an IELTS coaching WhatsApp support agent. Built with Python 3.11+, Astral `uv`, Google Gemini embeddings (`gemini-embedding-001`, 3072d), Qdrant vector search, and asyncpg + SQLAlchemy 2.0 async persistence.

---

## Architecture & Directory Layout

```
.
├── .env.example              # Template for local & cloud environment variables
├── .gitignore                # Git ignores for venvs, secrets, cache & raw data
├── pyproject.toml            # uv project definition (Python >= 3.11)
├── README.md                 # Project documentation
├── data/
│   └── raw_docs/             # Storage for downloaded source PDFs
└── src/
    ├── config.py             # pydantic-settings config with secret masking & Supabase docs
    ├── db/
    │   ├── postgres.py       # asyncpg + SQLAlchemy 2.0 async engine & message_history table
    │   └── vector_store.py   # Qdrant client factory (:memory: vs URL, 3072d auto-recreate)
    ├── ingestion/
    │   ├── pdf_loader.py     # Async httpx download + pypdf text extraction
    │   ├── chunker.py        # Custom recursive character splitter
    │   └── embedder.py       # Google Gemini embeddings + Qdrant upsert & similarity search
    └── main.py               # CLI pipeline runner with rich terminal feedback
```

---

## Features

- **Astral `uv` Package Management**: Fast, reproducible dependency resolution pinned for Python 3.11+.
- **Google Gemini Embeddings**: Powered by Google's `google-genai` SDK using `gemini-embedding-001` (3072 dimensions).
- **Zero-Dependency Vector DB Local Mode**: Uses Qdrant's embedded `:memory:` mode by default. Switching to a cloud Qdrant cluster requires only setting `QDRANT_URL` and `QDRANT_API_KEY`. Automatically detects and migrates stale collection vector dimensions.
- **Async PostgreSQL Persistence**: Uses SQLAlchemy 2.0 declarative models with `asyncpg` for non-blocking I/O. Ready for Supabase via `SUPABASE_DB_URL`.
- **Custom Recursive Text Splitter**: Independent of LangChain. Respects paragraph (`\n\n`), line (`\n`), and sentence (`. `, `? `, `! `) boundaries.
- **Modern Terminal Diagnostics**: Rich terminal interface with masked secrets, progress tracking, query inspection, and subsystem health check tables.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (Astral package manager)

### 2. Install Project Dependencies
```bash
uv sync
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```ini
# Gemini API Key (from Google AI Studio: https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Vector DB:
# - Leave blank for local in-memory execution:
QDRANT_URL=
QDRANT_API_KEY=
# - For cloud/remote Qdrant:
# QDRANT_URL=https://xyz.cloud.qdrant.io:6333
# QDRANT_API_KEY=your-api-key

# Supabase PostgreSQL Async Database URL (Transaction Pooler - port 6543):
# NOTE: Convert 'postgresql://' -> 'postgresql+asyncpg://', percent-encode special characters
# in the password (e.g. '@' -> '%40', '#' -> '%23'), and append '?ssl=require'.
SUPABASE_DB_URL=postgresql+asyncpg://postgres.your-project-ref:your_encoded_password@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres?ssl=require
```

---

## Running the Ingestion CLI

Run the full end-to-end pipeline with the default sample PDF:
```bash
uv run python src/main.py
```

Or provide a custom PDF URL:
```bash
uv run python src/main.py --pdf-url "https://raw.githubusercontent.com/py-pdf/pypdf/main/resources/crazyones.pdf"
```

To run offline with mock deterministic vectors (without calling the live Gemini API):
```bash
uv run python src/main.py --offline
```
