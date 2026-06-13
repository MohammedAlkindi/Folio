import logging
from typing import Any

from core.config import Config
from ingestion.embedder import embed_texts, load_model
from store.vector import get_all_chunks, query as vector_query

logger = logging.getLogger(__name__)

# Model cache keyed by model name so different configurations each get their own instance.
_model_cache: dict[str, Any] = {}


def _get_model(cfg: Config) -> Any:
    model_name = cfg.embedding_model()
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]


def retrieve(
    question: str,
    cfg: Config,
    top_k: int | None = None,
) -> list[dict]:
    """
    Embed question, query ChromaDB for semantic results, and (when hybrid is enabled)
    fuse with BM25 results via Reciprocal Rank Fusion.
    Each result: {text, filename, page_number, chunk_index, distance}.
    Never raises — returns [] on failure.
    """
    k = top_k if top_k is not None else cfg.retrieval_top_k()
    workspace = cfg.workspace()
    try:
        model = _get_model(cfg)
        embeddings = embed_texts([question], model)
        if not embeddings:
            logger.warning("Embedding returned empty result for question")
            return []
        semantic_results = vector_query(embedding=embeddings[0], top_k=k, workspace=workspace)
        logger.debug("Retrieved %d chunks for question (semantic)", len(semantic_results))

        if not cfg.retrieval_hybrid():
            return semantic_results

        from retrieval.bm25 import BM25Index  # noqa: PLC0415
        from retrieval.fusion import reciprocal_rank_fusion  # noqa: PLC0415

        all_chunks = get_all_chunks(workspace=workspace)
        if not all_chunks:
            return semantic_results

        bm25_index = BM25Index(all_chunks)
        bm25_results = bm25_index.search(question, top_k=k)
        fused = reciprocal_rank_fusion(semantic_results, bm25_results)
        logger.debug("Hybrid retrieval produced %d fused results", len(fused))
        return fused[:k]

    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []
