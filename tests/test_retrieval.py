"""Tests for hybrid retrieval: BM25Index, RRF fusion, and combined retrieve()."""
from unittest.mock import MagicMock, patch

import pytest

from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion


# ── BM25 ─────────────────────────────────────────────────────────────────────

def test_bm25_returns_results_ranked_by_keyword_relevance():
    chunks = [
        {"text": "the quick brown fox jumped", "filename": "a.txt", "page_number": 1, "chunk_index": 0},
        {"text": "machine learning neural networks deep learning", "filename": "b.txt", "page_number": 1, "chunk_index": 0},
        {"text": "fox ran across the meadow", "filename": "c.txt", "page_number": 1, "chunk_index": 0},
    ]
    index = BM25Index(chunks)
    results = index.search("fox", top_k=2)

    assert len(results) == 2
    filenames = [r["filename"] for r in results]
    # Both fox-containing chunks should rank above the ML chunk.
    assert "b.txt" not in filenames
    assert "a.txt" in filenames or "c.txt" in filenames


def test_bm25_result_shape():
    chunks = [
        {"text": "hello world", "filename": "x.txt", "page_number": 2, "chunk_index": 3},
    ]
    index = BM25Index(chunks)
    results = index.search("hello", top_k=1)

    assert len(results) == 1
    r = results[0]
    assert r["filename"] == "x.txt"
    assert r["page_number"] == 2
    assert r["chunk_index"] == 3
    assert "distance" in r
    assert 0.0 <= r["distance"] <= 1.0


def test_bm25_empty_index():
    index = BM25Index([])
    assert index.search("anything", top_k=5) == []


def test_bm25_top_k_respected():
    chunks = [
        {"text": f"document number {i}", "filename": f"f{i}.txt", "page_number": 1, "chunk_index": i}
        for i in range(10)
    ]
    index = BM25Index(chunks)
    results = index.search("document", top_k=3)
    assert len(results) <= 3


# ── RRF fusion ───────────────────────────────────────────────────────────────

def test_rrf_deduplicates_and_reranks():
    semantic = [
        {"text": "t1", "filename": "a.txt", "page_number": 1, "chunk_index": 0},
        {"text": "t2", "filename": "b.txt", "page_number": 1, "chunk_index": 0},
    ]
    bm25 = [
        {"text": "t2", "filename": "b.txt", "page_number": 1, "chunk_index": 0},  # duplicate
        {"text": "t3", "filename": "c.txt", "page_number": 1, "chunk_index": 0},
    ]
    fused = reciprocal_rank_fusion(semantic, bm25)

    keys = [(r["filename"], r["chunk_index"]) for r in fused]
    # b.txt appears in both lists — should appear exactly once.
    assert keys.count(("b.txt", 0)) == 1
    assert len(fused) == 3
    # b.txt ranks in both → highest combined RRF score → first position.
    assert fused[0]["filename"] == "b.txt"


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
    semantic = [{"text": "t", "filename": "a.txt", "page_number": 1, "chunk_index": 0}]
    result = reciprocal_rank_fusion(semantic, [])
    assert len(result) == 1
    assert result[0]["filename"] == "a.txt"


def test_rrf_dedup_key_is_filename_and_chunk_index():
    # Same filename, different chunk_index → NOT a duplicate.
    semantic = [{"text": "a", "filename": "doc.txt", "page_number": 1, "chunk_index": 0}]
    bm25 = [{"text": "b", "filename": "doc.txt", "page_number": 2, "chunk_index": 1}]
    fused = reciprocal_rank_fusion(semantic, bm25)
    assert len(fused) == 2


# ── Hybrid retrieve() ─────────────────────────────────────────────────────────

def test_hybrid_retrieve_combines_results(cfg):
    """retrieve() should call vector search and BM25 and return fused results."""
    semantic = [
        {"text": "the quick brown fox", "filename": "x.pdf", "page_number": 1,
         "chunk_index": 0, "distance": 0.1},
    ]
    all_chunks = [
        {"text": "the quick brown fox", "filename": "x.pdf", "page_number": 1, "chunk_index": 0},
        {"text": "unrelated content here", "filename": "y.pdf", "page_number": 1, "chunk_index": 0},
    ]

    with patch("retrieval.retriever.embed_texts", return_value=[[0.1, 0.2]]), \
         patch("retrieval.retriever.vector_query", return_value=semantic), \
         patch("retrieval.retriever.get_all_chunks", return_value=all_chunks), \
         patch("retrieval.retriever._get_model", return_value=MagicMock()):
        from retrieval.retriever import retrieve
        results = retrieve("fox", cfg)

    assert len(results) >= 1
    filenames = [r["filename"] for r in results]
    assert "x.pdf" in filenames


def test_hybrid_disabled_returns_semantic_only(cfg):
    """When retrieval.hybrid is False, BM25 is skipped."""
    from core.config import Config
    cfg_no_hybrid = Config(
        {
            **cfg._data,
            "retrieval": {"top_k": 3, "hybrid": False},
        }
    )
    semantic = [
        {"text": "semantic only", "filename": "a.txt", "page_number": 1,
         "chunk_index": 0, "distance": 0.05},
    ]

    with patch("retrieval.retriever.embed_texts", return_value=[[0.1]]), \
         patch("retrieval.retriever.vector_query", return_value=semantic) as mock_vq, \
         patch("retrieval.retriever.get_all_chunks") as mock_all, \
         patch("retrieval.retriever._get_model", return_value=MagicMock()):
        from retrieval.retriever import retrieve
        results = retrieve("query", cfg_no_hybrid)

    mock_all.assert_not_called()
    assert results == semantic
