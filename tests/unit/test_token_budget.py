from repochat.retrieval.token_budget import fit_to_budget


class _FakeChunk:
    def __init__(self, token_count: int):
        self.token_count = token_count


def test_fits_all_chunks_under_budget():
    chunks = [_FakeChunk(100), _FakeChunk(100), _FakeChunk(100)]
    result = fit_to_budget(chunks, budget=1000)
    assert len(result) == 3


def test_drops_low_ranked_chunks_over_budget():
    chunks = [_FakeChunk(400), _FakeChunk(400), _FakeChunk(400)]
    result = fit_to_budget(chunks, budget=900)
    assert len(result) == 2


def test_always_includes_at_least_one_chunk_even_if_it_alone_exceeds_budget():
    chunks = [_FakeChunk(5000)]
    result = fit_to_budget(chunks, budget=1000)
    assert len(result) == 1


def test_empty_input():
    assert fit_to_budget([], budget=1000) == []
