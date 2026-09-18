from repochat.retrieval.score_fusion import combine_scores, min_max_normalize


def test_min_max_normalize_scales_to_zero_one():
    result = min_max_normalize([2.0, 4.0, 6.0])
    assert result == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_all_tied_values():
    result = min_max_normalize([5.0, 5.0, 5.0])
    assert result == [0.5, 0.5, 0.5]


def test_min_max_normalize_empty():
    assert min_max_normalize([]) == []


def test_combine_scores_respects_dominant_weight():
    # rerank dominates at 0.70 -- the top-reranked candidate should win even
    # if it's mid-pack on RRF and graph support.
    rerank_scores = [8.2, 1.0, 0.5]
    rrf_scores = [0.01, 0.016, 0.005]
    graph_scores = [0.0, 0.9, 0.0]
    combined = combine_scores(rerank_scores, rrf_scores, graph_scores)
    assert combined.index(max(combined)) == 0


def test_combine_scores_without_normalization_would_be_dominated_by_widest_range():
    # Sanity check on the premise: raw (unnormalized) values here would let
    # the graph score's 0-0.9 range swing the result more than intended,
    # since it's numerically larger than the RRF scores. Normalizing fixes it.
    rerank_scores = [1.0, 1.0]
    rrf_scores = [0.016, 0.016]
    graph_scores = [0.0, 0.9]
    combined = combine_scores(rerank_scores, rrf_scores, graph_scores)
    # rerank and rrf are tied (normalize to 0.5 each), so only the graph
    # weight (0.10) should separate them -- a small, bounded difference.
    assert abs(combined[1] - combined[0]) <= 0.10 + 1e-9
