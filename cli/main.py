import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from folio.core.config import load_config
from folio.core.manifest import (
    file_ingested,
    file_skipped,
    ingest_complete,
    ingest_start,
    query_logged,
)
from folio.core.paths import ensure_dirs
from folio.ingestion.chunker import chunk_pages
from folio.ingestion.embedder import embed_texts, load_model
from folio.ingestion.parser import parse
from folio.ingestion.scanner import doc_id, scan_folder
from folio.qa.answerer import answer
from folio.retrieval.retriever import retrieve
from folio.store.db import (
    document_exists,
    init_db,
    insert_chunk,
    insert_document,
    list_documents,
    update_chunk_count,
)
from folio.store.models import Chunk, Document
from folio.store.vector import upsert_chunks

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

        click.echo(f"  Ingesting  {filepath.name}…", nl=False)

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

        # Persist to ChromaDB.
        upsert_chunks(chunk_objs, embeddings)

        total_chunks += len(chunk_objs)
        new_count += 1
        file_ingested(str(filepath), len(chunk_objs), "ok")
        click.echo(f"  OK ({len(chunk_objs)} chunks)")

    ingest_complete(new_count, total_chunks)

    click.echo("")
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

        click.echo(f"\n{result['answer']}\n")

        sources = result.get("sources", [])
        if sources:
            source_parts = []
            for s in sources:
                page = s.get("page_number")
                label = f"{s['filename']} (p.{page})" if page else s["filename"]
                source_parts.append(label)
            click.echo(f"Sources: {', '.join(source_parts)}")

        confidence = result.get("confidence", "unknown")
        click.echo(f"Confidence: {confidence}\n")

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
        pages_label = str(doc.page_count) if doc.page_count is not None else "—"
        # ISO timestamp → date only for compact display.
        date_label = doc.ingested_at[:10]
        click.echo(
            f"{doc.filename:<40} {pages_label:>6}  {doc.chunk_count:>7}  {date_label}"
        )


if __name__ == "__main__":
    cli()
