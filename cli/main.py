import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from core.config import load_config
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
from ingestion.scanner import doc_id, scan_folder
from qa.answerer import answer
from retrieval.retriever import retrieve
from store.db import (
    delete_document,
    document_exists,
    get_document_by_filename,
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


@click.group()
def cli() -> None:
    """Folio — local document intelligence pipeline."""


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

    # Load embedding model once for the whole batch.
    model = load_model(cfg.embedding_model())

    new_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks = 0

    for filepath in files:
        fid = doc_id(filepath)

        if document_exists(fid):
            skipped_count += 1
            file_skipped(str(filepath), "already indexed")
            logger.info("SKIP  %s (already indexed)", filepath.name)
            continue

        click.echo(f"  Ingesting  {filepath.name}...", nl=False)

        pages = parse(filepath)
        if not pages:
            failed_count += 1
            file_skipped(str(filepath), "no extractable text")
            logger.warning("FAIL  %s — no extractable text", filepath.name)
            click.echo("  FAILED (no text)")
            continue

        raw_chunks = chunk_pages(pages, cfg.chunk_size(), cfg.chunk_overlap())
        if not raw_chunks:
            failed_count += 1
            file_skipped(str(filepath), "chunking produced no output")
            click.echo("  FAILED (no chunks)")
            continue

        # Build Chunk objects.
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

        # Embed all chunks for this document.
        embeddings = embed_texts([c.text for c in chunk_objs], model)

        # Determine page count from PDF pages (None for txt/md).
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
        )

        # Persist to SQLite.
        insert_document(doc)
        for chunk in chunk_objs:
            insert_chunk(chunk)
        update_chunk_count(fid, len(chunk_objs))

        # Persist to ChromaDB under the configured workspace.
        upsert_chunks(chunk_objs, embeddings, workspace=workspace)

        total_chunks += len(chunk_objs)
        new_count += 1
        file_ingested(str(filepath), len(chunk_objs), "ok")
        click.echo(f"  OK ({len(chunk_objs)} chunks)")

    ingest_complete(new_count, total_chunks)

    click.echo("")
    click.echo(f"Workspace: {workspace}")
    click.echo(f"Scanned:   {len(files):>6} files")
    click.echo(f"Ingested:  {new_count:>6} new")
    click.echo(f"Skipped:   {skipped_count:>6} (already indexed)")
    click.echo(f"Failed:    {failed_count:>6}")
    click.echo(f"Chunks:    {total_chunks:>6,} total")


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

        # Confidence-gated warning prefix (Dimension 5).
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

        # Source excerpt preview (Dimension 5).
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
def list_docs(config_path: str) -> None:
    """Print a table of all ingested documents."""
    # Config is loaded to validate the file exists; not strictly needed for list.
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

    for doc in docs:
        pages_label = str(doc.page_count) if doc.page_count is not None else "-"
        # ISO timestamp -> date only for compact display.
        date_label = doc.ingested_at[:10]
        click.echo(
            f"{doc.filename:<40} {pages_label:>6}  {doc.chunk_count:>7}  {date_label}"
        )


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

    # Step 1 — remove existing data.
    delete_by_doc_id(doc.id, workspace=workspace)
    delete_document(doc.id)
    click.echo(f"Removed existing index entry for '{filename}'.")

    if not filepath.exists():
        click.echo(f"Error: original file no longer exists at '{filepath}'.", err=True)
        sys.exit(1)

    # Step 2 — re-ingest.
    click.echo(f"  Re-ingesting  {filepath.name}...", nl=False)

    pages = parse(filepath)
    if not pages:
        click.echo("  FAILED (no text)")
        sys.exit(1)

    raw_chunks = chunk_pages(pages, cfg.chunk_size(), cfg.chunk_overlap())
    if not raw_chunks:
        click.echo("  FAILED (no chunks)")
        sys.exit(1)

    model = load_model(cfg.embedding_model())
    fid = doc_id(filepath)

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

    new_doc = Document(
        id=fid,
        filename=filepath.name,
        filepath=str(filepath.resolve()),
        extension=filepath.suffix.lower(),
        page_count=page_count,
        chunk_count=len(chunk_objs),
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

    insert_document(new_doc)
    for chunk in chunk_objs:
        insert_chunk(chunk)
    update_chunk_count(fid, len(chunk_objs))
    upsert_chunks(chunk_objs, embeddings, workspace=workspace)

    click.echo(f"  OK ({len(chunk_objs)} chunks)")
    file_ingested(str(filepath), len(chunk_objs), "reindex")


if __name__ == "__main__":
    cli()
