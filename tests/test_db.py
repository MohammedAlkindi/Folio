"""Tests for store/db.py — all tests use tmp_db fixture, never data/folio.db."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

import store.db as db_module
from store.models import Chunk, Document


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_doc(suffix: str = "") -> Document:
    return Document(
        id=f"docid_{suffix}",
        filename=f"test{suffix}.txt",
        filepath=f"/tmp/test{suffix}.txt",
        extension=".txt",
        page_count=None,
        chunk_count=0,
        ingested_at=_now_iso(),
    )


def _make_chunk(doc_id: str, index: int = 0) -> Chunk:
    return Chunk(
        id=f"{doc_id}:chunk:{index}",
        doc_id=doc_id,
        filename="test.txt",
        page_number=None,
        chunk_index=index,
        text=f"chunk text {index}",
        token_estimate=3,
    )


# ---------------------------------------------------------------------------
# Fixture: redirect DB_PATH to tmp_db for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_db: Path, monkeypatch):
    """Redirect all db_module calls to the temporary database path.

    This guarantees that no test ever reads or writes data/folio.db.
    Assumption: db_module.DB_PATH is the single source of truth for the path
    used by _conn(), so patching it here is sufficient.
    """
    monkeypatch.setattr(db_module, "DB_PATH", tmp_db)
    # Also patch the imported name inside core.paths that db_module re-imports.
    import core.paths as paths_module
    monkeypatch.setattr(paths_module, "DB_PATH", tmp_db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_db():
    db_module.init_db()  # must not raise


def test_insert_and_exists():
    db_module.init_db()
    doc = _make_doc("a")
    db_module.insert_document(doc)
    assert db_module.document_exists(doc.id)


def test_idempotent_insert():
    db_module.init_db()
    doc = _make_doc("b")
    first = db_module.insert_document(doc)
    second = db_module.insert_document(doc)
    assert first is True
    assert second is False  # duplicate insert returns False, no error


def test_list_documents():
    db_module.init_db()
    doc1 = _make_doc("c1")
    doc2 = _make_doc("c2")
    db_module.insert_document(doc1)
    db_module.insert_document(doc2)
    docs = db_module.list_documents()
    ids = {d.id for d in docs}
    assert doc1.id in ids
    assert doc2.id in ids


def test_insert_chunk():
    db_module.init_db()
    doc = _make_doc("d")
    db_module.insert_document(doc)
    chunk = _make_chunk(doc.id, index=0)
    db_module.insert_chunk(chunk)
    retrieved = db_module.get_chunks_for_doc(doc.id)
    assert len(retrieved) == 1
    assert retrieved[0].chunk_index == 0


def test_delete_document():
    db_module.init_db()
    doc = _make_doc("e")
    db_module.insert_document(doc)
    db_module.insert_chunk(_make_chunk(doc.id, 0))
    db_module.insert_chunk(_make_chunk(doc.id, 1))

    db_module.delete_document(doc.id)

    assert not db_module.document_exists(doc.id)
    assert db_module.get_chunks_for_doc(doc.id) == []


def test_get_document_by_filename():
    db_module.init_db()
    doc = _make_doc("f")
    db_module.insert_document(doc)
    found = db_module.get_document_by_filename(doc.filename)
    assert found is not None
    assert found.id == doc.id


def test_get_document_by_filename_missing():
    db_module.init_db()
    result = db_module.get_document_by_filename("does_not_exist.txt")
    assert result is None


def test_update_chunk_count():
    db_module.init_db()
    doc = _make_doc("g")
    db_module.insert_document(doc)
    db_module.update_chunk_count(doc.id, 42)
    updated = db_module.get_document(doc.id)
    assert updated.chunk_count == 42
