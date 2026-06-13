import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core.config import Config, load_config
from core.manifest import (
    file_ingested,
    file_skipped,
    ingest_complete,
    ingest_start,
    query_logged,
)
from core.paths import ensure_dirs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_texts, load_model
from ingestion.parser import parse
from ingestion.scanner import content_hash, doc_id, scan_folder
from qa.answerer import answer
from retrieval.retriever import retrieve
from store.db import (
    delete_document,
    document_exists,
    get_chunks_for_doc,
    get_document_by_filename,
    get_document_by_filepath,
    init_db,
    insert_chunk,
    insert_document,
    list_documents,
    update_chunk_count,
)
from store.models import Chunk, Document
from store.vector import delete_by_doc_id, list_workspaces, upsert_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("folio")

_PARSE_REASON_LABELS: dict[str, str] = {
    "encrypted":   "FAILED (encrypted PDF — decrypt before ingesting)",
    "corrupted":   "FAILED (corrupted or malformed PDF)",
    "no_text":     "FAILED (no extractable text — possibly scanned)",
    "unsupported": "FAILED (unsupported file type)",
    "parse_error": "FAILED (parse error — see logs)",
}


@click.group()
def cli() -> None:
    """Folio — local document intelligence pipeline."""


# ── shared pipeline helper ────────────────────────────────────────────────────

def _ingest_single_file(filepath: Path, cfg, model, workspace: str) -> tuple[int, str]:
    """
    Run the full ingest pipeline for one file.

    If the file is already indexed (same doc_id) it is deleted first (re-index path).
    Returns (chunk_count, status) where status is one of:
      "ingested" | "reindexed" | "failed:<reason>"
    """
    fid = doc_id(filepath)
    is_reindex = document_exists(fid)

    if is_reindex:
        delete_by_doc_id(fid, workspace=workspace)
        delete_document(fid)

    pages, parse_reason = parse(filepath)
    if not pages:
        file_skipped(str(filepath), parse_reason or "no_text")
        logger.warning("FAIL  %s — %s", filepath.name, parse_reason or "no_text")
        return (0, f"failed:{parse_reason or 'no_text'}")

    raw_chunks = chunk_pages(pages, cfg.chunk_size(), cfg.chunk_overlap())
    if not raw_chunks:
        file_skipped(str(filepath), "chunking produced no output")
        return (0, "failed:no_text")

    chunk_objs: list[Chunk] = [
        Chunk(
            id=f"{fid}:chunk:{rc['chunk_index']}",
            doc_id=fid,
            filename=filepath.name,
            page_number=rc["page_number"],
            chunk_index=rc["chunk_index"],
            text=rc["text"],
            token_estimate=len(rc["text"].split()),
        )
        for rc in raw_chunks
    ]

    embeddings = embed_texts([c.text for c in chunk_objs], model)

    page_numbers = [p["page_number"] for p in pages if p["page_number"] is not None]
    page_count = max(page_numbers) if page_numbers else None

    doc = Document(
        id=fid,
        filename=filepath.name,
        filepath=str(filepath.resolve()),
        extension=filepath.suffix.lower(),
        page_count=page_count,
        chunk_count=len(chunk_objs),
        ingested_at=datetime.now(timezone.utc).isoformat(),
        content_hash=content_hash(filepath),
    )

    insert_document(doc)
    for chunk in chunk_objs:
        insert_chunk(chunk)
    update_chunk_count(fid, len(chunk_objs))
    upsert_chunks(chunk_objs, embeddings, workspace=workspace)

    status = "reindexed" if is_reindex else "ingested"
    file_ingested(str(filepath), len(chunk_objs), status)
    return (len(chunk_objs), status)


# ── watch event handler ───────────────────────────────────────────────────────

class FolioEventHandler(FileSystemEventHandler):
    """Watchdog handler that ingests or removes files as they change."""

    def __init__(self, cfg, model, workspace: str, extensions: set) -> None:
        super().__init__()
        self._cfg = cfg
        self._model = model
        self._workspace = workspace
        self._extensions = extensions

    def _matches(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._extensions

    def on_created(self, event) -> None:
        if event.is_directory or not self._matches(event.src_path):
            return
        filepath = Path(event.src_path)
        chunk_count, status = _ingest_single_file(filepath, self._cfg, self._model, self._workspace)
        if not status.startswith("failed:"):
            click.echo(f"  [+] Ingested {filepath.name} ({chunk_count} chunks)")
        else:
            reason = status.split(":", 1)[1]
            label = _PARSE_REASON_LABELS.get(reason, "unknown error")
            click.echo(f"  [!] {filepath.name} — {label}")

    def on_modified(self, event) -> None:
        if event.is_directory or not self._matches(event.src_path):
            return
        filepath = Path(event.src_path)

        # Skip if file content has not changed since last ingest.
        new_ch = content_hash(filepath)
        existing = get_document_by_filepath(str(filepath.resolve()))
        if existing is not None and existing.content_hash == new_ch:
            return

        if existing is not None:
            delete_by_doc_id(existing.id, workspace=self._workspace)
            delete_document(existing.id)

        chunk_count, status = _ingest_single_file(filepath, self._cfg, self._model, self._workspace)
        if not status.startswith("failed:"):
            click.echo(f"  [~] Auto-reindexed {filepath.name} (content changed) ({chunk_count} chunks)")
        else:
            reason = status.split(":", 1)[1]
            label = _PARSE_REASON_LABELS.get(reason, "unknown error")
            click.echo(f"  [!] {filepath.name} — {label}")

    def on_deleted(self, event) -> None:
        if event.is_directory or not self._matches(event.src_path):
            return
        filepath = Path(event.src_path)
        # Use filepath lookup so we don't need to read the (already deleted) file for its hash.
        existing = get_document_by_filepath(str(filepath.resolve()))
        if existing is None:
            return
        delete_by_doc_id(existing.id, workspace=self._workspace)
        delete_document(existing.id)
        click.echo(f"  [-] Removed {filepath.name}")


# ── ingest ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config/folio_config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
def ingest(config_path: str) -> None:
    """Scan a folder and ingest all supported documents into the local store."""
    cfg = load_config(config_path)
    ensure_dirs()
    init_db()

    folder = cfg.ingestion_folder()
    extensions = cfg.supported_extensions()
    workspace = cfg.workspace()

    ingest_start(folder)
    files = scan_folder(folder, extensions)

    if not files:
        click.echo(f"No supported files found in '{folder}'. Nothing to ingest.")
        return

    model = load_model(cfg.embedding_model())

    new_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks = 0

    for filepath in files:
        new_ch = content_hash(filepath)
        existing = get_document_by_filepath(str(filepath.resolve()))

        if existing is not None:
            if existing.content_hash is not None and existing.content_hash == new_ch:
                skipped_count += 1
                file_skipped(str(filepath), "already indexed")
                logger.info("SKIP  %s (already indexed)", filepath.name)
                continue

            # Content changed (or legacy entry without content_hash) — auto-reindex.
            delete_by_doc_id(existing.id, workspace=workspace)
            delete_document(existing.id)
            chunk_count, status = _ingest_single_file(filepath, cfg, model, workspace)
            if status.startswith("failed:"):
                reason = status.split(":", 1)[1]
                label = _PARSE_REASON_LABELS.get(reason, "FAILED (unknown error)")
                failed_count += 1
                click.echo(f"  {label}")
            else:
                total_chunks += chunk_count
                new_count += 1
                click.echo(
                    f"  [~] Auto-reindexed {filepath.name} (content changed) ({chunk_count} chunks)"
                )
            continue

        click.echo(f"  Ingesting  {filepath.name}...", nl=False)

        chunk_count, status = _ingest_single_file(filepath, cfg, model, workspace)

        if status.startswith("failed:"):
            reason = status.split(":", 1)[1]
            label = _PARSE_REASON_LABELS.get(reason, "FAILED (unknown error)")
            failed_count += 1
            click.echo(f"  {label}")
            continue

        total_chunks += chunk_count
        new_count += 1
        click.echo(f"  OK ({chunk_count} chunks)")

    ingest_complete(new_count, total_chunks)

    click.echo("")
    click.echo(f"Workspace: {workspace}")
    click.echo(f"Scanned:   {len(files):>6} files")
    click.echo(f"Ingested:  {new_count:>6} new")
    click.echo(f"Skipped:   {skipped_count:>6} (already indexed)")
    click.echo(f"Failed:    {failed_count:>6}")
    click.echo(f"Chunks:    {total_chunks:>6,} total")


# ── watch ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config/folio_config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
def watch(config_path: str) -> None:
    """Monitor the ingestion folder and auto-ingest files on change."""
    cfg = load_config(config_path)
    ensure_dirs()
    init_db()

    folder = cfg.ingestion_folder()
    extensions = {ext.lower() for ext in cfg.supported_extensions()}
    workspace = cfg.workspace()

    model = load_model(cfg.embedding_model())

    click.echo(f"Watching {folder} for changes. Press Ctrl+C to stop.")

    handler = FolioEventHandler(cfg, model, workspace, extensions)
    observer = Observer()
    observer.schedule(handler, str(folder), recursive=True)
    observer.start()

    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        click.echo("Stopped.")


# ── demo ─────────────────────────────────────────────────────────────────────

_DEMO_QUESTIONS = [
    "What revenue targets were approved?",
    "Who attended the board meeting?",
    "What was the Q4 hiring plan?",
    "What risks were flagged by the CFO?",
]

_DEMO_CONFIG_BASE: dict = {
    "workspace": "demo",
    "ingestion": {
        "folder": "",  # filled at runtime
        "supported_extensions": [".pdf", ".txt", ".md"],
        "chunk_size": 800,
        "chunk_overlap": 150,
    },
    "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
    "retrieval": {"top_k": 6},
    "qa": {
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 2048,
        "context_budget": 3500,
    },
}

_DEMO_PDF = "board_minutes_q3_2024.pdf"


@cli.command()
@click.option("--reset", is_flag=True, default=False, help="Clear the demo index and re-ingest from scratch.")
def demo(reset: bool) -> None:
    """Ingest the built-in sample PDF and start an interactive Q&A session.

    Requires ANTHROPIC_API_KEY to be set. No config file needed.
    Re-run with --reset to clear the demo index and start fresh.
    """
    # ── pre-flight checks ────────────────────────────────────────────────────

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        click.echo(
            "Error: ANTHROPIC_API_KEY is not set.\n\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n\n"
            "Folio needs this key to answer questions. Ingestion works without it,\n"
            "but the Q&A session will fail when you ask the first question.",
            err=True,
        )
        sys.exit(1)

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    sample_pdf = docs_dir / _DEMO_PDF

    if not sample_pdf.exists():
        click.echo(
            f"Error: {_DEMO_PDF} not found in docs/.\n\n"
            "Make sure you are running from the repo root and the file was not deleted.\n"
            "To recreate it: see the generation script in the project history.",
            err=True,
        )
        sys.exit(1)

    config_data = {
        **_DEMO_CONFIG_BASE,
        "ingestion": {**_DEMO_CONFIG_BASE["ingestion"], "folder": str(docs_dir)},
    }
    cfg = Config(config_data)

    ensure_dirs()
    init_db()

    # ── header ───────────────────────────────────────────────────────────────

    click.echo("=" * 60)
    click.echo("  Folio Demo — Meridian Capital Q3 2024 Board Meeting")
    click.echo("=" * 60)

    # ── ingest ───────────────────────────────────────────────────────────────

    fid = doc_id(sample_pdf)
    already_indexed = document_exists(fid)

    if already_indexed and reset:
        click.echo("  --reset: clearing demo index...")
        delete_by_doc_id(fid, workspace=cfg.workspace())
        delete_document(fid)
        already_indexed = False

    if already_indexed:
        click.echo(f"  {_DEMO_PDF} already indexed  (use --reset to re-ingest)")
    else:
        click.echo("  Loading embedding model...", nl=False)
        try:
            model = load_model(cfg.embedding_model())
        except Exception as exc:
            click.echo(f"\nError: could not load embedding model — {exc}", err=True)
            sys.exit(1)
        click.echo("  OK")

        click.echo(f"  Ingesting {_DEMO_PDF}...", nl=False)
        chunk_count, status = _ingest_single_file(sample_pdf, cfg, model, cfg.workspace())
        if status.startswith("failed:"):
            reason = status.split(":", 1)[1]
            click.echo(f"\n  {_PARSE_REASON_LABELS.get(reason, 'FAILED (unknown error)')}", err=True)
            sys.exit(1)
        click.echo(f"  OK ({chunk_count} chunks)")

    # ── query loop ───────────────────────────────────────────────────────────

    click.echo("")
    click.echo("  Suggested questions:")
    for q in _DEMO_QUESTIONS:
        click.echo(f"    - {q}")
    click.echo("")
    click.echo("  Type 'quit' to exit.\n")

    while True:
        try:
            question = click.prompt(">", prompt_suffix=" ").strip()
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            click.echo("Goodbye.")
            break

        chunks = retrieve(question, cfg)
        if not chunks:
            click.echo("No relevant content found. Try rephrasing your question.\n")
            continue

        try:
            result = answer(question, chunks, cfg)
        except Exception as exc:
            click.echo(f"Error: Q&A request failed — {exc}\n", err=True)
            continue

        confidence = result.get("confidence", "unknown")

        if confidence == "insufficient_data":
            click.echo("\nWarning: Folio found limited evidence. The answer below may be incomplete.")
        elif confidence == "low":
            click.echo("\nNote: Low confidence — consider ingesting more relevant documents.")

        click.echo(f"\n{result['answer']}\n")

        sources = result.get("sources", [])
        if sources:
            parts = []
            for s in sources:
                page = s.get("page_number")
                parts.append(f"{s['filename']} (p.{page})" if page else s["filename"])
            click.echo(f"Sources: {', '.join(parts)}")

        click.echo(f"Confidence: {confidence}\n")

        try:
            show = click.prompt("Show excerpts? [y/N]", default="N", show_default=False).strip().lower()
        except (click.Abort, EOFError, KeyboardInterrupt):
            show = "n"

        if show in {"y", "yes"}:
            click.echo("")
            for i, chunk in enumerate(chunks, start=1):
                page = chunk.get("page_number")
                click.echo(f"--- [{i}] {chunk['filename']} ({'p.' + str(page) if page else 'no page'}) ---")
                click.echo(chunk["text"])
                click.echo("")

        query_logged(question, [s.get("filename", "") for s in sources])


# ── query ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config/folio_config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
def query(config_path: str) -> None:
    """Interactive Q&A loop over the ingested document corpus."""
    cfg = load_config(config_path)
    ensure_dirs()
    init_db()

    click.echo("Folio — ask questions about your documents. Type 'quit' to exit.\n")

    while True:
        try:
            question = click.prompt(">", prompt_suffix=" ").strip()
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            click.echo("Goodbye.")
            break

        chunks = retrieve(question, cfg)
        if not chunks:
            click.echo("No relevant content found. Have you run 'folio ingest' yet?\n")
            continue

        result = answer(question, chunks, cfg)

        confidence = result.get("confidence", "unknown")

        if confidence == "insufficient_data":
            click.echo(
                "\nWarning: Folio did not find strong evidence in your documents. "
                "The following answer may be incomplete."
            )
        elif confidence == "low":
            click.echo(
                "\nNote: Low confidence — consider ingesting more relevant documents."
            )

        click.echo(f"\n{result['answer']}\n")

        sources = result.get("sources", [])
        if sources:
            source_parts = []
            for s in sources:
                page = s.get("page_number")
                label = f"{s['filename']} (p.{page})" if page else s["filename"]
                source_parts.append(label)
            click.echo(f"Sources: {', '.join(source_parts)}")

        click.echo(f"Confidence: {confidence}\n")

        if chunks:
            try:
                show = click.prompt("Show excerpts? [y/N]", default="N", show_default=False).strip().lower()
            except (click.Abort, EOFError, KeyboardInterrupt):
                show = "n"

            if show in {"y", "yes"}:
                click.echo("")
                for i, chunk in enumerate(chunks, start=1):
                    page = chunk.get("page_number")
                    page_label = f"p.{page}" if page else "no page"
                    click.echo(f"--- [{i}] {chunk['filename']} ({page_label}) ---")
                    click.echo(chunk["text"])
                    click.echo("")

        source_names = [s.get("filename", "") for s in sources]
        query_logged(question, source_names)


# ── list ─────────────────────────────────────────────────────────────────────

@cli.command(name="list")
@click.option(
    "--config",
    "config_path",
    default="config/folio_config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show chunk previews.")
def list_docs(config_path: str, verbose: bool) -> None:
    """Print a table of all ingested documents."""
    load_config(config_path)
    ensure_dirs()
    init_db()

    docs = list_documents()
    if not docs:
        click.echo("No documents ingested yet. Run 'folio ingest' first.")
        return

    header = f"{'Filename':<40} {'Pages':>6}  {'Chunks':>7}  {'Ingested'}"
    click.echo(header)
    click.echo("-" * len(header))

    _PREVIEW_LIMIT = 5

    for doc in docs:
        pages_label = str(doc.page_count) if doc.page_count is not None else "-"
        date_label = doc.ingested_at[:10]
        click.echo(
            f"{doc.filename:<40} {pages_label:>6}  {doc.chunk_count:>7}  {date_label}"
        )
        if verbose:
            chunks = get_chunks_for_doc(doc.id)
            for chunk in chunks[:_PREVIEW_LIMIT]:
                page_label = str(chunk.page_number) if chunk.page_number is not None else "-"
                preview = chunk.preview or "(no preview)"
                click.echo(f"    [{chunk.chunk_index}] p.{page_label}: {preview}")
            remainder = len(chunks) - _PREVIEW_LIMIT
            if remainder > 0:
                click.echo(f"    ... and {remainder} more chunk{'s' if remainder > 1 else ''}")


# ── workspace ────────────────────────────────────────────────────────────────

@cli.group()
def workspace() -> None:
    """Manage named document workspaces."""


@workspace.command(name="list")
def workspace_list() -> None:
    """List all ChromaDB collections (workspaces) with document counts."""
    ensure_dirs()
    workspaces = list_workspaces()
    if not workspaces:
        click.echo("No workspaces found. Run 'folio ingest' to create one.")
        return

    header = f"{'Workspace':<30} {'Collection':<35} {'Chunks':>7}"
    click.echo(header)
    click.echo("-" * len(header))
    for w in workspaces:
        click.echo(f"{w['workspace']:<30} {w['collection']:<35} {w['count']:>7}")


# ── remove ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("filename")
@click.option(
    "--config",
    "config_path",
    default="config/folio_config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
def remove(filename: str, config_path: str) -> None:
    """Remove a document from SQLite and ChromaDB by filename."""
    cfg = load_config(config_path)
    ensure_dirs()
    init_db()

    doc = get_document_by_filename(filename)
    if doc is None:
        click.echo(f"Error: '{filename}' not found in the index.", err=True)
        sys.exit(1)

    workspace = cfg.workspace()
    chunk_count = doc.chunk_count

    delete_by_doc_id(doc.id, workspace=workspace)
    delete_document(doc.id)

    click.echo(f"Removed: {filename} ({chunk_count} chunks deleted)")


# ── reindex ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("filename")
@click.option(
    "--config",
    "config_path",
    default="config/folio_config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
def reindex(filename: str, config_path: str) -> None:
    """Remove and re-ingest a document by filename."""
    cfg = load_config(config_path)
    ensure_dirs()
    init_db()

    doc = get_document_by_filename(filename)
    if doc is None:
        click.echo(f"Error: '{filename}' not found in the index.", err=True)
        sys.exit(1)

    filepath = Path(doc.filepath)
    workspace = cfg.workspace()

    delete_by_doc_id(doc.id, workspace=workspace)
    delete_document(doc.id)
    click.echo(f"Removed existing index entry for '{filename}'.")

    if not filepath.exists():
        click.echo(f"Error: original file no longer exists at '{filepath}'.", err=True)
        sys.exit(1)

    model = load_model(cfg.embedding_model())
    click.echo(f"  Re-ingesting  {filepath.name}...", nl=False)

    # File was already deleted above, so _ingest_single_file will take the fresh-ingest path.
    chunk_count, status = _ingest_single_file(filepath, cfg, model, workspace)

    if status.startswith("failed:"):
        reason = status.split(":", 1)[1]
        label = _PARSE_REASON_LABELS.get(reason, "FAILED (unknown error)")
        click.echo(f"  {label}")
        sys.exit(1)

    click.echo(f"  OK ({chunk_count} chunks)")


if __name__ == "__main__":
    cli()
