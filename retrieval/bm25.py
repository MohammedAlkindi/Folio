import logging
from typing import Any

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 keyword search index over a list of chunk dicts."""

    def __init__(self, chunks: list[dict]) -> None:
        """
        Build the index from chunk dicts.
        Expected keys: text, filename, page_number, chunk_index.
        """
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for hybrid search: pip install rank-bm25"
            )

        self._chunks = chunks
        if not chunks:
            self._bm25: Any = None
            return
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[dict]:
        """
        Return top_k results sorted by BM25 score descending.
        Output shape matches store/vector.py query():
        {text, filename, page_number, chunk_index, distance}
        where distance is in [0, 1] (0 = best match, 1 = worst).
        """
        if not self._chunks or self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # BM25Okapi can return negative scores when a term appears in all documents
        # (negative IDF). Normalise using min-max so distance is always in [0, 1].
        # Assumption: when all scores are equal (score_range == 0), all chunks match
        # equally well — assign distance=0 (best) to all returned results.
        min_score = float(min(scores))
        max_score = float(max(scores))
        score_range = max_score - min_score

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for idx, score in indexed[:top_k]:
            normalised = (float(score) - min_score) / score_range if score_range > 0 else 1.0
            results.append(
                {
                    "text": self._chunks[idx]["text"],
                    "filename": self._chunks[idx]["filename"],
                    "page_number": self._chunks[idx].get("page_number"),
                    "chunk_index": self._chunks[idx]["chunk_index"],
                    # distance convention: 0 = perfect match, 1 = worst match.
                    "distance": 1.0 - normalised,
                }
            )
        return results
