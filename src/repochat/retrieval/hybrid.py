"""Reciprocal rank fusion: merge lexical + dense ranked lists.

Their scores live on incompatible scales (a BM25 score and a cosine
distance aren't comparable), so ranks are fused, not raw scores -- the
design doc calls this out as the first robust merge to reach for (6.2).
"""

RRF_K = 60


def reciprocal_rank_fusion(*ranked_lists: list, k: int = RRF_K) -> list[tuple[object, float]]:
    """Each ranked_list is items in rank order (best first). Returns
    (item, fused_score) pairs sorted best-first. Items compare by identity,
    which holds here because dense and lexical results both come from
    SQLAlchemy's identity map within the same session."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    session,
    revision_id,
    *,
    query: str,
    query_vector: list[float],
    lexical_index,
    dense_k: int = 30,
    lexical_k: int = 30,
    top_k: int = 15,
) -> list:
    from repochat.retrieval.dense_retriever import search_dense

    dense_chunks = search_dense(session, revision_id, query_vector, top_k=dense_k)
    lexical_chunks = lexical_index.search(query, top_k=lexical_k)

    fused = reciprocal_rank_fusion(dense_chunks, lexical_chunks)
    return [chunk for chunk, _ in fused[:top_k]]
