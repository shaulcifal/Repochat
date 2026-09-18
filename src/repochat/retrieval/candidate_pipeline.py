"""Phase 3 + 4 candidate pipeline: hybrid retrieval -> file-level
diversification -> controlled 1-hop import-graph expansion.

This is what answer_service calls instead of raw dense or hybrid search.
"""

from repochat.domain.models import Chunk
from repochat.retrieval.file_scoring import score_and_diversify
from repochat.retrieval.graph_expansion import expand_via_imports
from repochat.retrieval.hybrid import hybrid_search

CANDIDATE_POOL_SIZE = 30


def retrieve_candidates(
    session,
    revision_id,
    *,
    query: str,
    query_vector: list[float],
    lexical_index,
    top_k: int = 8,
) -> list[Chunk]:
    hybrid_chunks = hybrid_search(
        session,
        revision_id,
        query=query,
        query_vector=query_vector,
        lexical_index=lexical_index,
        dense_k=CANDIDATE_POOL_SIZE,
        lexical_k=CANDIDATE_POOL_SIZE,
        top_k=CANDIDATE_POOL_SIZE,
    )
    lexical_only = lexical_index.search(query, top_k=CANDIDATE_POOL_SIZE)

    diversified = score_and_diversify(hybrid_chunks, lexical_only)
    expanded = expand_via_imports(session, revision_id, diversified, query_vector)

    seen_ids = {chunk.id for chunk in diversified}
    expanded_unique = [chunk for chunk in expanded if chunk.id not in seen_ids]

    # Reserve room for expanded evidence rather than letting the primary
    # diversified list (which alone can already reach top_k) crowd it out --
    # otherwise graph expansion would never actually reach the answer.
    reserved_for_expansion = min(len(expanded_unique), top_k // 2)
    primary = diversified[: top_k - reserved_for_expansion]
    return primary + expanded_unique[: top_k - len(primary)]
