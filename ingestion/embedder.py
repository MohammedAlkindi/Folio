import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


def load_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load and return a sentence-transformers model. Expensive — call once and cache."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        )
    logger.info("Loading embedding model '%s'…", model_name)
    t0 = time.monotonic()
    model = SentenceTransformer(model_name)
    logger.info("Model loaded in %.2fs", time.monotonic() - t0)
    return model


def embed_texts(texts: list[str], model: Any) -> list[list[float]]:
    """
    Embed texts in batches of _BATCH_SIZE.
    Returns a list of float vectors in the same order as input.
    Assumption: sentence-transformers encode() returns numpy arrays; we convert to list[float].
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        t0 = time.monotonic()
        vecs = model.encode(batch, convert_to_numpy=True)
        elapsed = time.monotonic() - t0
        logger.debug(
            "Embedded batch [%d:%d] (%d texts) in %.2fs",
            batch_start,
            batch_start + len(batch),
            len(batch),
            elapsed,
        )
        all_embeddings.extend(v.tolist() for v in vecs)

    return all_embeddings
