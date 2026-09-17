import argparse
import asyncio
import io
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure src root is in python path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import get_settings
from db.postgres import get_async_engine, log_rag_interaction, verify_postgres_read_write
from db.vector_store import ensure_collection, get_qdrant_client
from ingestion.chunker import RecursiveTextSplitter, TextChunk
from ingestion.embedder import (
    GeminiEmbedder,
    generate_deterministic_mock_vector,
    query_qdrant_similar,
    upsert_chunks_to_qdrant,
)
from ingestion.pdf_loader import download_pdf, extract_text_from_pdf
from rag.engine import answer_question

console = Console(legacy_windows=False)

# Reliable, small public PDF for testing document ingestion
DEFAULT_SAMPLE_PDF_URL = "https://raw.githubusercontent.com/py-pdf/pypdf/main/resources/crazyones.pdf"


def render_config_table(settings_summary: Dict[str, str]) -> None:
    """Render a masked configuration summary table using Rich."""
    table = Table(
        title="[bold yellow]⚙️  IELTS Spirit - Configuration Summary[/bold yellow]",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Parameter", style="bold white", width=24)
    table.add_column("Configured Value / Mode", style="green")

    for key, value in settings_summary.items():
        table.add_row(key, value)

    console.print(table)
    console.print()


def render_subsystems_table(statuses: Dict[str, Dict[str, Any]]) -> None:
    """Render the final subsystem status table with checkmarks/crosses."""
    table = Table(
        title="[bold blue]📊 Subsystem Verification Status[/bold blue]",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Subsystem", style="bold white", width=30)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Details", style="dim white")

    for name, info in statuses.items():
        status_icon = "[bold green]✅ PASS[/bold green]" if info["ok"] else "[bold red]❌ FAIL[/bold red]"
        table.add_row(name, status_icon, info["detail"])

    console.print(table)
    console.print()


async def run_pipeline(pdf_url: str, allow_fake_embeddings: bool = False) -> None:
    """Run full ingestion, chunking, embedding, upsert, retrieval test, and Postgres check."""
    settings = get_settings()

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]IELTS Spirit - Ingestion & Persistence Engine[/bold cyan]\n"
            "[dim]RAG-Powered WhatsApp Support System Foundation[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    if allow_fake_embeddings:
        console.print(
            Panel(
                "[bold red]⚠ WARNING: RUNNING WITH FAKE / OFFLINE EMBEDDINGS FLAG ENABLED[/bold red]\n"
                "[yellow]No live Gemini API calls will be made. Mock deterministic vectors will be used.[/yellow]",
                border_style="red",
            )
        )
        console.print()

    # Step 1: Display config summary
    render_config_table(settings.masked_summary())

    subsystem_status: Dict[str, Dict[str, Any]] = {
        "Gemini Embeddings": {"ok": False, "detail": "Pending verification"},
        "Qdrant Vector DB": {"ok": False, "detail": "Pending verification"},
        "PostgreSQL (Supabase Async)": {"ok": False, "detail": "Pending verification"},
    }

    # Step 2: Download & Parse PDF
    docs_dir = PROJECT_ROOT / "data" / "raw_docs"
    console.print(f"[bold blue]Step 1/4:[/bold blue] Fetching PDF from [underline]{pdf_url}[/underline]...")
    try:
        pdf_path = await download_pdf(pdf_url, docs_dir)
        console.print(f"  [green]✔[/green] Downloaded to: [bold]{pdf_path.relative_to(PROJECT_ROOT)}[/bold]")

        pages = extract_text_from_pdf(pdf_path)
        total_chars = sum(len(p["text"]) for p in pages)
        console.print(f"  [green]✔[/green] Extracted {len(pages)} page(s) ({total_chars} total characters).")
    except Exception as exc:
        console.print(f"  [bold red]✖ Failed to load PDF:[/bold red] {exc}")
        console.print("[bold red]Pipeline halted due to document load failure.[/bold red]")
        sys.exit(1)

    # Step 3: Chunking
    console.print("\n[bold blue]Step 2/4:[/bold blue] Chunking text with recursive character splitter...")
    splitter = RecursiveTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks: List[TextChunk] = splitter.split_pages(pages)
    console.print(
        f"  [green]✔[/green] Generated [bold]{len(chunks)}[/bold] chunk(s) "
        f"(chunk_size={settings.chunk_size} chars, overlap={settings.chunk_overlap} chars)."
    )
    if chunks:
        sample_preview = chunks[0].text[:80].replace("\n", " ")
        console.print(f"  [dim]Sample chunk (chars={chunks[0].char_count}): \"{sample_preview}...\"[/dim]")

    # Step 4: Embedding generation
    console.print("\n[bold blue]Step 3/4:[/bold blue] Generating embeddings & indexing into Qdrant...")
    chunk_embeddings: List[List[float]] = []

    if allow_fake_embeddings:
        subsystem_status["Gemini Embeddings"] = {
            "ok": True,
            "detail": f"OFFLINE MODE: Deterministic mock vectors ({settings.embedding_dimension}d)",
        }
        chunk_embeddings = [
            generate_deterministic_mock_vector(c.text, settings.embedding_dimension) for c in chunks
        ]
        console.print("  [yellow]ℹ Generated deterministic mock vectors for offline run.[/yellow]")
    else:
        try:
            embedder = GeminiEmbedder(
                api_key=settings.gemini_api_key,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
            )
            chunk_texts = [c.text for c in chunks]
            chunk_embeddings = await embedder.get_embeddings(chunk_texts)
            actual_dim = len(chunk_embeddings[0]) if chunk_embeddings else settings.embedding_dimension

            subsystem_status["Gemini Embeddings"] = {
                "ok": True,
                "detail": f"Model: '{settings.embedding_model}' (dim={actual_dim})",
            }
            console.print(
                f"  [green]✔[/green] Gemini embedding API verified successfully. "
                f"Generated {len(chunk_embeddings)} vector(s) of dimension {actual_dim}."
            )
        except Exception as exc:
            subsystem_status["Gemini Embeddings"] = {
                "ok": False,
                "detail": f"Embed call failed: {exc}",
            }
            console.print(f"\n[bold red]✖ CRITICAL: Gemini embedding call failed:[/bold red] {exc}")
            console.print("[red]Silent fake-vector fallback has been removed. Halting execution.[/red]")
            console.print("[dim]Use --offline or --fake-embeddings if you explicitly require mock vectors.[/dim]")
            sys.exit(1)

    # Qdrant client & collection
    qdrant_client = None
    try:
        qdrant_client = get_qdrant_client(settings)
        ensure_collection(qdrant_client, settings.qdrant_collection_name, settings.embedding_dimension)

        upserted_count = upsert_chunks_to_qdrant(
            client=qdrant_client,
            collection_name=settings.qdrant_collection_name,
            chunks=chunks,
            embeddings=chunk_embeddings,
        )
        mode_desc = "Local Storage (data/qdrant_storage)" if settings.is_qdrant_in_memory else f"Remote ({settings.qdrant_url})"
        subsystem_status["Qdrant Vector DB"] = {
            "ok": True,
            "detail": f"{mode_desc} | Collection: '{settings.qdrant_collection_name}' ({upserted_count} points, {settings.embedding_dimension}d)",
        }
        console.print(f"  [green]✔[/green] Successfully upserted {upserted_count} vector(s) into Qdrant [{mode_desc}].")
    except Exception as exc:
        subsystem_status["Qdrant Vector DB"] = {
            "ok": False,
            "detail": f"Qdrant error: {exc}",
        }
        console.print(f"  [bold red]✖ Qdrant operation failed:[/bold red] {exc}")

    # Run test similarity query
    console.print("\n[bold blue]Step 4/4:[/bold blue] Executing test similarity retrieval...")
    test_query = "Who are the rebels, troublemakers, and crazy ones that change things?"
    console.print(f"  [italic]Query:[/italic] [bold cyan]\"{test_query}\"[/bold cyan]")

    if qdrant_client:
        try:
            if allow_fake_embeddings:
                query_vector = generate_deterministic_mock_vector(test_query, settings.embedding_dimension)
            else:
                query_vector = await embedder.get_query_embedding(test_query)

            search_results = query_qdrant_similar(
                client=qdrant_client,
                collection_name=settings.qdrant_collection_name,
                query_vector=query_vector,
                limit=3,
            )

            results_table = Table(
                title="[bold green]🔍 Top Similarity Matches[/bold green]",
                box=box.SIMPLE_HEAD,
                header_style="bold cyan",
            )
            results_table.add_column("Rank", justify="center", width=6)
            results_table.add_column("Score", justify="right", width=12)
            results_table.add_column("Page", justify="center", width=8)
            results_table.add_column("Chunk Snippet", style="white")

            for rank, hit in enumerate(search_results, start=1):
                payload = hit.payload or {}
                text_preview = payload.get("text", "")[:120].replace("\n", " ") + "..."
                page_num = str(payload.get("page", 1))
                score_str = f"[bold green]{hit.score:.4f}[/bold green]" if hit.score >= 0.3 else f"[yellow]{hit.score:.4f}[/yellow]"
                results_table.add_row(str(rank), score_str, page_num, text_preview)

            console.print(results_table)
            console.print()
        except Exception as exc:
            console.print(f"  [bold red]✖ Similarity query failed:[/bold red] {exc}")

    # PostgreSQL / Supabase async connectivity & read/write verification
    console.print("[bold blue]Checking Database Persistence:[/bold blue] PostgreSQL / Supabase async connection...")
    if not settings.supabase_db_url:
        subsystem_status["PostgreSQL (Supabase Async)"] = {
            "ok": False,
            "detail": "SUPABASE_DB_URL is not set in environment.",
        }
        console.print("  [yellow]⚠ SUPABASE_DB_URL not set in environment. Skipping database connection.[/yellow]")
    else:
        try:
            engine = get_async_engine(settings.supabase_db_url)
            ok, detail, fetched_row, full_tb = await verify_postgres_read_write(engine)
            subsystem_status["PostgreSQL (Supabase Async)"] = {
                "ok": ok,
                "detail": detail,
            }
            if ok and fetched_row is not None:
                console.print(f"  [green]✔[/green] {detail}")
                console.print(f"  [bold cyan]✔ Verified Readback Row:[/bold cyan] [green]{fetched_row}[/green]")
            else:
                console.print(f"  [bold red]✖ PostgreSQL verification failed:[/bold red] {detail}")
                if full_tb:
                    console.print("\n[bold red]── FULL TRACEBACK ──────────────────────────────────────────[/bold red]")
                    console.print(full_tb)
                    console.print("[bold red]────────────────────────────────────────────────────────────[/bold red]\n")
            await engine.dispose()
        except Exception as exc:
            import traceback
            full_tb = traceback.format_exc()
            subsystem_status["PostgreSQL (Supabase Async)"] = {
                "ok": False,
                "detail": f"Engine connection error: {exc}",
            }
            console.print(f"  [bold red]✖ Database verification error:[/bold red] {exc}")
            console.print("\n[bold red]── FULL TRACEBACK ──────────────────────────────────────────[/bold red]")
            console.print(full_tb)
            console.print("[bold red]────────────────────────────────────────────────────────────[/bold red]\n")

    console.print()
    render_subsystems_table(subsystem_status)


async def run_ask_pipeline(question: str) -> None:
    """Run RAG query against already-indexed Qdrant collection and log to Supabase PostgreSQL."""
    settings = get_settings()

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]IELTS Spirit - RAG Query Engine[/bold cyan]\n"
            f"[dim]Question:[/dim] [bold white]\"{question}\"[/bold white]",
            border_style="cyan",
        )
    )
    console.print()

    # Step 1: Display config summary
    render_config_table(settings.masked_summary())

    # Step 2: Answer question via RAG engine
    console.print(f"[bold blue]Step 1/2:[/bold blue] Searching Qdrant and generating grounded answer...")
    qdrant_client = get_qdrant_client(settings)

    rag_result = await answer_question(
        query=question,
        client=qdrant_client,
        settings=settings,
        top_k=3,
        similarity_threshold=0.5,
    )
    qdrant_client.close()

    # Display retrieved chunks
    chunks = rag_result.get("chunks", [])
    scores = rag_result.get("scores", [])

    if chunks:
        chunks_table = Table(
            title="[bold green]🔍 Retrieved Context Chunks from Qdrant[/bold green]",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
        )
        chunks_table.add_column("Rank", justify="center", width=6)
        chunks_table.add_column("Similarity Score", justify="center", width=18)
        chunks_table.add_column("Page", justify="center", width=8)
        chunks_table.add_column("Content Snippet", style="white")

        for idx, (chunk, score) in enumerate(zip(chunks, scores), start=1):
            text = chunk.get("text", "").replace("\n", " ")
            snippet = text[:140] + "..." if len(text) > 140 else text
            page_str = str(chunk.get("page", "?"))
            score_badge = f"[bold green]{score:.4f}[/bold green]" if score >= 0.5 else f"[yellow]{score:.4f}[/yellow]"
            chunks_table.add_row(str(idx), score_badge, page_str, snippet)

        console.print(chunks_table)
        console.print()
    else:
        console.print("  [yellow]⚠ No context chunks passed the similarity threshold (>= 0.5).[/yellow]\n")

    # Display Generated Answer
    model_name = rag_result.get("model_used") or "None"
    answer_text = rag_result.get("answer", "")
    console.print(
        Panel(
            f"[bold white]{answer_text}[/bold white]",
            title=f"[bold green]💬 Generated Grounded Answer (Model: {model_name})[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    # Step 3: Log to Supabase PostgreSQL message_history
    console.print("[bold blue]Step 2/2:[/bold blue] Logging exchange to Supabase PostgreSQL [bold](message_history)[/bold]...")
    if not settings.supabase_db_url:
        console.print("  [yellow]⚠ SUPABASE_DB_URL is not set. Skipping chat history persistence.[/yellow]\n")
        return

    try:
        engine = get_async_engine(settings.supabase_db_url)
        logged_ok, rows, log_err = await log_rag_interaction(
            engine=engine,
            phone_number="+10000000000",
            question=question,
            answer=answer_text,
        )
        if logged_ok and rows:
            db_table = Table(
                title="[bold cyan]💾 Persisted Rows in Supabase (message_history)[/bold cyan]",
                box=box.DOUBLE_EDGE,
                header_style="bold cyan",
                show_lines=True,
            )
            db_table.add_column("ID", justify="center", width=8, style="bold magenta")
            db_table.add_column("Phone", width=16)
            db_table.add_column("Role", justify="center", width=12)
            db_table.add_column("Content Preview", style="white", min_width=35)
            db_table.add_column("Created At (UTC)", width=24, style="dim white")

            for r in rows:
                role_badge = "[cyan]user[/cyan]" if r.role == "user" else "[green]assistant[/green]"
                preview = r.content.replace("\n", " ")
                if len(preview) > 90:
                    preview = preview[:90] + "..."
                db_table.add_row(
                    str(r.id),
                    r.phone_number,
                    role_badge,
                    preview,
                    str(r.created_at),
                )

            console.print("  [green]✔[/green] Both question and answer were successfully persisted and verified!")
            console.print(db_table)
        else:
            console.print(f"  [yellow]⚠ Warning: Could not log to message_history: {log_err}[/yellow]")

        await engine.dispose()
    except Exception as exc:
        console.print(f"  [yellow]⚠ Warning: Error during PostgreSQL logging (non-fatal): {exc}[/yellow]")

    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="IELTS Spirit - Ingestion & Persistence CLI")
    parser.add_argument(
        "--pdf-url",
        type=str,
        default=None,
        help="Public URL of the PDF document to ingest into Qdrant",
    )
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        help="Question to ask the RAG engine against indexed content",
    )
    parser.add_argument(
        "--offline",
        "--fake-embeddings",
        action="store_true",
        dest="offline",
        help="Explicit flag to run with mock deterministic vectors (offline mode)",
    )
    args = parser.parse_args()

    if args.ask:
        asyncio.run(run_ask_pipeline(args.ask))
    else:
        pdf_target = args.pdf_url or DEFAULT_SAMPLE_PDF_URL
        asyncio.run(run_pipeline(pdf_target, allow_fake_embeddings=args.offline))


if __name__ == "__main__":
    main()
