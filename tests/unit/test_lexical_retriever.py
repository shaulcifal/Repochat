from repochat.domain.models import Chunk
from repochat.retrieval.lexical_retriever import LexicalIndex


def _chunk(qualified_name: str, embedding_text: str, tags: str = "") -> Chunk:
    # Unpersisted ORM objects are fine here -- LexicalIndex only reads
    # plain attributes, no session/DB involved.
    return Chunk(
        language="python",
        symbol_kind="function",
        qualified_name=qualified_name,
        parent_symbol=None,
        start_line=1,
        end_line=5,
        signature=f"def {qualified_name}():",
        docstring=None,
        tags=tags,
        raw_source="pass",
        embedding_text=embedding_text,
        token_count=10,
        content_hash="x",
        parse_status="ok",
        embedding=[0.0] * 4,
    )


def test_exact_identifier_ranks_top():
    chunks = [
        _chunk("save_checkpoint", "[symbol: save_checkpoint] persists model state to disk", tags="save checkpoint"),
        _chunk("load_checkpoint", "[symbol: load_checkpoint] restores model state from disk", tags="load checkpoint"),
        _chunk("autodetect_device_type", "[symbol: autodetect_device_type] picks cuda or cpu", tags="autodetect device type"),
    ]
    index = LexicalIndex(chunks)
    results = index.search("save_checkpoint", top_k=3)
    assert results
    assert results[0].qualified_name == "save_checkpoint"


def test_no_match_returns_empty():
    chunks = [_chunk("save_checkpoint", "[symbol: save_checkpoint] persists model state", tags="save checkpoint")]
    index = LexicalIndex(chunks)
    results = index.search("zzz_completely_unrelated_query_xyz", top_k=3)
    assert results == []
