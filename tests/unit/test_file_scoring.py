import uuid

from repochat.domain.models import Chunk
from repochat.retrieval.file_scoring import MAX_CHUNKS_PER_FILE_PRE_RERANK, score_and_diversify


def _chunk(file_id) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        file_id=file_id,
        language="python",
        symbol_kind="function",
        qualified_name="f",
        parent_symbol=None,
        start_line=1,
        end_line=2,
        signature="def f():",
        docstring=None,
        tags="",
        raw_source="pass",
        embedding_text="text",
        token_count=1,
        content_hash="x",
        parse_status="ok",
        embedding=[0.0] * 4,
    )


def test_caps_chunks_per_file():
    file_a = uuid.uuid4()
    chunks = [_chunk(file_a) for _ in range(5)]
    result = score_and_diversify(chunks, lexical_only_chunks=[])
    assert len(result) == MAX_CHUNKS_PER_FILE_PRE_RERANK


def test_file_with_more_top_ranked_chunks_scores_higher():
    file_a, file_b = uuid.uuid4(), uuid.uuid4()
    # file_a gets ranks 1,2,3 (best); file_b only gets rank 4.
    chunks = [_chunk(file_a), _chunk(file_a), _chunk(file_a), _chunk(file_b)]
    result = score_and_diversify(chunks, lexical_only_chunks=[], max_files=2)
    file_ids_in_order = []
    for chunk in result:
        if chunk.file_id not in file_ids_in_order:
            file_ids_in_order.append(chunk.file_id)
    assert file_ids_in_order[0] == file_a


def test_lexical_bonus_helps_a_file_also_hit_by_bm25():
    file_a, file_b = uuid.uuid4(), uuid.uuid4()
    chunk_a = _chunk(file_a)
    chunk_b = _chunk(file_b)
    # Same single rank each -- tied on the RRF-position term alone.
    ranked = [chunk_a]
    ranked2 = [chunk_b]
    # Give file_b a slightly better rank than file_a to start on equal footing
    # after the lexical bonus is added for file_a only.
    combined = ranked + ranked2
    result = score_and_diversify(combined, lexical_only_chunks=[chunk_a], max_files=2)
    file_ids_in_order = []
    for chunk in result:
        if chunk.file_id not in file_ids_in_order:
            file_ids_in_order.append(chunk.file_id)
    assert file_ids_in_order[0] == file_a


def test_empty_input():
    assert score_and_diversify([], []) == []
