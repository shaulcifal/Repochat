"""The full Phase 3-5 retrieval pipeline, instrumented to produce a
RetrievalTrace: lexical -> dense -> RRF -> diversify -> graph expand ->
rerank -> normalize/combine -> token budget. One call produces both the
final chunk list and a JSON-serializable trace of every stage (design doc
Fix 6), so a bad answer is debuggable stage-by-stage.
"""

from dataclasses import dataclass

from repochat.domain.models import Chunk
from repochat.retrieval.dense_retriever import search_dense
from repochat.retrieval.file_scoring import score_and_diversify
from repochat.retrieval.graph_expansion import expand_via_imports
from repochat.retrieval.hybrid import reciprocal_rank_fusion
from repochat.retrieval.reranker import RerankerProvider
from repochat.retrieval.scope import RetrievalPolicy, Scope, policy_for
from repochat.retrieval.score_fusion import combine_scores
from repochat.retrieval.token_budget import fit_to_budget

GRAPH_EDGE_CONFIDENCE = 0.9  # matches indexing/pipeline.py's IMPORT_EDGE_CONFIDENCE


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    trace: dict  # JSON-serializable


def _chunk_ref(chunk: Chunk) -> dict:
    return {"chunk_id": str(chunk.id), "path": chunk.file.path, "symbol": chunk.qualified_name}


def run_retrieval(
    session,
    revision_id,
    *,
    query: str,
    query_vector: list[float],
    lexical_index,
    reranker: RerankerProvider,
    policy: RetrievalPolicy | None = None,
) -> RetrievalResult:
    # No classifier available (or none wanted) -> the previous fixed behavior.
    policy = policy or policy_for(Scope.FEATURE)

    dense_chunks = search_dense(session, revision_id, query_vector, top_k=policy.candidate_pool_size)
    lexical_chunks = lexical_index.search(query, top_k=policy.candidate_pool_size)

    fused_pairs = reciprocal_rank_fusion(dense_chunks, lexical_chunks)
    rrf_score_by_id = {chunk.id: score for chunk, score in fused_pairs}
    ranked_chunks = [chunk for chunk, _ in fused_pairs]

    diversified = score_and_diversify(ranked_chunks, lexical_chunks, max_files=policy.max_files)
    expanded = expand_via_imports(
        session, revision_id, diversified, query_vector, max_expanded_files=policy.max_expanded_files
    )
    seen_ids = {chunk.id for chunk in diversified}
    expanded_unique = [chunk for chunk in expanded if chunk.id not in seen_ids]

    candidates = diversified + expanded_unique
    graph_confidence_by_id = {chunk.id: 0.0 for chunk in diversified}
    graph_confidence_by_id.update({chunk.id: GRAPH_EDGE_CONFIDENCE for chunk in expanded_unique})

    rerank_scores = reranker.score(query, [c.embedding_text for c in candidates])
    rrf_scores = [rrf_score_by_id.get(c.id, 0.0) for c in candidates]
    graph_scores = [graph_confidence_by_id.get(c.id, 0.0) for c in candidates]
    final_scores = combine_scores(rerank_scores, rrf_scores, graph_scores)

    ranked_final = [
        chunk for chunk, _ in sorted(zip(candidates, final_scores), key=lambda pair: pair[1], reverse=True)
    ]
    # Budget over a slightly larger pool than top_k so token limits, not just
    # chunk count, decide what gets trimmed.
    budgeted = fit_to_budget(ranked_final[: policy.top_k * 2])
    final = budgeted[: policy.top_k]

    trace = {
        "scope": policy.scope.value,
        "lexical_results": [_chunk_ref(c) for c in lexical_chunks],
        "dense_results": [_chunk_ref(c) for c in dense_chunks],
        "rrf_results": [{"chunk_id": str(c.id), "score": s} for c, s in fused_pairs],
        "graph_expansion": [_chunk_ref(c) for c in expanded_unique],
        "reranker_results": [
            {
                "chunk_id": str(c.id),
                "rerank_raw": r,
                "rrf_raw": rr,
                "graph_raw": g,
                "final": f,
            }
            for c, r, rr, g, f in zip(candidates, rerank_scores, rrf_scores, graph_scores, final_scores)
        ],
        "final_chunks": [_chunk_ref(c) for c in final],
        "context_chunks": [_chunk_ref(c) for c in final],
    }
    return RetrievalResult(chunks=final, trace=trace)
