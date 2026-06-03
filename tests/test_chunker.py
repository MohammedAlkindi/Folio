"""Tests for ingestion/chunker.py — no external dependencies required."""
from ingestion.chunker import chunk_pages


def _make_pages(word_count: int, page_number: int = 1) -> list[dict]:
    """Return a single-page pages list with exactly word_count words."""
    words = " ".join(f"word{i}" for i in range(word_count))
    return [{"page_number": page_number, "text": words}]


def test_basic_chunking():
    pages = _make_pages(200)
    chunks = chunk_pages(pages, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert "page_number" in c
        assert "chunk_index" in c


def test_overlap():
    pages = _make_pages(120)
    chunks = chunk_pages(pages, chunk_size=50, overlap=20)
    assert len(chunks) >= 2
    # Adjacent chunks must share some words at the boundary.
    words_first = set(chunks[0]["text"].split())
    words_second = set(chunks[1]["text"].split())
    overlap_words = words_first & words_second
    assert len(overlap_words) > 0, "Adjacent chunks should share words at overlap boundary"


def test_single_chunk():
    # Text shorter than chunk_size must produce exactly one chunk.
    pages = _make_pages(30)
    chunks = chunk_pages(pages, chunk_size=50, overlap=10)
    assert len(chunks) == 1


def test_empty_input():
    assert chunk_pages([]) == []


def test_chunk_index_sequential():
    pages = _make_pages(300)
    chunks = chunk_pages(pages, chunk_size=60, overlap=10)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index values must be 0,1,2,... with no gaps"
