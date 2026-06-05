import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from core.paths import DB_PATH, ensure_dirs
from store.models import Chunk, Document

logger = logging.getLogger(__name__)

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    extension   TEXT NOT NULL,
    page_count  INTEGER,
    chunk_count INTEGER DEFAULT 0,
    ingested_at TEXT NOT NULL
);
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(id),
    filename       TEXT NOT NULL,
    page_number    INTEGER,
    chunk_index    INTEGER NOT NULL,
    token_estimate INTEGER,
    preview        TEXT,
    ingested_at    TEXT NOT NULL
);
"""


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    ensure_dirs()
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    # WAL mode allows concurrent readers alongside one writer.
    con.execute("PRAGMA journal_mode=WAL;")
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _conn_at(db_path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a connection to an arbitrary path — used by tests."""
    import sqlite3 as _sqlite3
    from contextlib import contextmanager as _cm

    @_cm
    def _inner():
        con = _sqlite3.connect(str(db_path), check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.row_factory = _sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    return _inner()


def init_db() -> None:
    with _conn() as con:
        con.execute(_CREATE_DOCUMENTS)
        con.execute(_CREATE_CHUNKS)
        # Migration: add preview column to existing databases that predate this field.
        try:
            con.execute("ALTER TABLE chunks ADD COLUMN preview TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
    logger.debug("SQLite schema initialised at %s", DB_PATH)


def document_exists(doc_id: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    return row is not None


def insert_document(doc: Document) -> bool:
    """Insert document. Returns True if new, False if already present."""
    if document_exists(doc.id):
        return False
    with _conn() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO documents
                (id, filename, filepath, extension, page_count, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.id,
                doc.filename,
                doc.filepath,
                doc.extension,
                doc.page_count,
                doc.chunk_count,
                doc.ingested_at,
            ),
        )
    return True


def update_chunk_count(doc_id: str, chunk_count: int) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE documents SET chunk_count = ? WHERE id = ?",
            (chunk_count, doc_id),
        )


def insert_chunk(chunk: Chunk) -> None:
    now = datetime.now(timezone.utc).isoformat()
    preview = chunk.text[:200] if chunk.text else ""
    with _conn() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO chunks
                (id, doc_id, filename, page_number, chunk_index, token_estimate, preview, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id,
                chunk.doc_id,
                chunk.filename,
                chunk.page_number,
                chunk.chunk_index,
                chunk.token_estimate,
                preview,
                now,
            ),
        )


def get_document(doc_id: str) -> Document | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    if row is None:
        return None
    return Document(**dict(row))


def get_document_by_filename(filename: str) -> Document | None:
    """Look up the first document matching the given filename."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM documents WHERE filename = ? LIMIT 1", (filename,)
        ).fetchone()
    if row is None:
        return None
    return Document(**dict(row))


def list_documents() -> list[Document]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
    return [Document(**dict(r)) for r in rows]


def get_chunks_for_doc(doc_id: str) -> list[Chunk]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        result.append(
            Chunk(
                id=d["id"],
                doc_id=d["doc_id"],
                filename=d["filename"],
                page_number=d["page_number"],
                chunk_index=d["chunk_index"],
                text="",  # full text lives in ChromaDB only
                token_estimate=d["token_estimate"],
                preview=d.get("preview") or "",
            )
        )
    return result


def delete_document(doc_id: str) -> None:
    """Cascade-delete document and its chunks."""
    with _conn() as con:
        con.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        con.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
