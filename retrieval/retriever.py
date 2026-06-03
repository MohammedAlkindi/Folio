import logging

from core.config import Config
from ingestion.embedder import embed_texts, load_model
from store.vector import query as vector_query

logger = logging.getLogger(__name__)

# Module-level model cache — loaded once per process on first query.
_model = None


def _get_model(cfg: Config):
    global _model
    if _model is None:
        _model = load_model(cfg.embedding_model())
    return _model


def retrieve(
    question: str,
    cfg: Config,
    top_k: int | None = None,
) -> list[dict]:
    """
    Embed question, query ChromaDB, return top_k ranked chunks.
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
        results = vector_query(embedding=embeddings[0], top_k=k, workspace=workspace)
        logger.debug("Retrieved %d chunks for question", len(results))
        return results
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []
