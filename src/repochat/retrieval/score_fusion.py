"""Normalize rerank/RRF/graph-support scores to a comparable range before
combining with fixed weights (design doc Fix 8).

Their raw scales are wildly different -- a reranker logit around 8, an RRF
score around 0.016, a graph confidence around 0.9 -- so combining them
directly would let whichever signal has the widest numeric range silently
dominate regardless of the weights.
"""


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def combine_scores(
    rerank_scores: list[float],
    rrf_scores: list[float],
    graph_scores: list[float],
    *,
    rerank_weight: float = 0.70,
    rrf_weight: float = 0.20,
    graph_weight: float = 0.10,
) -> list[float]:
    rerank_norm = min_max_normalize(rerank_scores)
    rrf_norm = min_max_normalize(rrf_scores)
    graph_norm = min_max_normalize(graph_scores)
    return [
        rerank_weight * r + rrf_weight * f + graph_weight * g
        for r, f, g in zip(rerank_norm, rrf_norm, graph_norm)
    ]
