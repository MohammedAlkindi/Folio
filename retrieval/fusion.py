def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    score(d) = sum_over_lists( 1 / (k + rank(d, list)) )

    Deduplication key: (filename, chunk_index).
    Returns a single list sorted by fused score descending.
    """
    scores: dict[tuple[str, int], float] = {}
    merged: dict[tuple[str, int], dict] = {}

    for rank, result in enumerate(semantic_results, start=1):
        key = (result["filename"], result["chunk_index"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        merged[key] = result

    for rank, result in enumerate(bm25_results, start=1):
        key = (result["filename"], result["chunk_index"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in merged:
            merged[key] = result

    sorted_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [merged[key] for key in sorted_keys]
