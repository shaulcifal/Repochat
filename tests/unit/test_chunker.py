import pytest

from repochat.chunking.chunker import _identifier_tags, _split_identifier_words, build_chunk_drafts
from repochat.parsers.python_ast import parse_python_source

SAMPLE = '''"""Module docstring."""
import os


def add(a, b):
    """Add two numbers."""
    return a + b
'''


def test_builds_module_and_function_chunks():
    parsed = parse_python_source(SAMPLE)
    drafts = build_chunk_drafts(parsed, repo_slug="acme/demo", path="src/math_utils.py", full_source=SAMPLE)
    kinds = [d.symbol_kind for d in drafts]
    assert "module" in kinds
    assert "function" in kinds


def test_function_chunk_has_correct_span_and_source():
    parsed = parse_python_source(SAMPLE)
    drafts = build_chunk_drafts(parsed, repo_slug="acme/demo", path="src/math_utils.py", full_source=SAMPLE)
    add_chunk = next(d for d in drafts if d.qualified_name == "add")
    assert add_chunk.start_line == 5
    assert add_chunk.end_line == 7
    assert "return a + b" in add_chunk.raw_source


def test_embedding_text_contains_metadata_header():
    parsed = parse_python_source(SAMPLE)
    drafts = build_chunk_drafts(parsed, repo_slug="acme/demo", path="src/math_utils.py", full_source=SAMPLE)
    add_chunk = next(d for d in drafts if d.qualified_name == "add")
    assert "[repository: acme/demo]" in add_chunk.embedding_text
    assert "[path: src/math_utils.py]" in add_chunk.embedding_text
    assert "[symbol: add]" in add_chunk.embedding_text
    assert "[docstring: Add two numbers.]" in add_chunk.embedding_text
    assert "SOURCE" in add_chunk.embedding_text


def test_partial_parse_falls_back_to_whole_file_chunk():
    parsed = parse_python_source("def broken(:\n    pass\n")
    drafts = build_chunk_drafts(parsed, repo_slug="acme/demo", path="src/broken.py", full_source="def broken(:\n    pass\n")
    assert len(drafts) == 1
    assert drafts[0].parse_status == "partial"
    assert drafts[0].symbol_kind == "module"


# ---------------------------------------------------------------------------
# Identifier enrichment: the tokens BM25 actually searches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier,expected",
    [
        # snake_case
        ("save_checkpoint", ["save", "checkpoint"]),
        ("kv_bytes_per_token", ["kv", "bytes", "per", "token"]),
        ("__init__", ["init"]),
        # camelCase / PascalCase
        ("camelCase", ["camel", "case"]),
        ("getResponse", ["get", "response"]),
        # acronym runs - these were all glued together before the
        # (?<=[A-Z])(?=[A-Z][a-z]) boundary was added
        ("RustBPETokenizer", ["rust", "bpe", "tokenizer"]),
        ("getHTTPResponse", ["get", "http", "response"]),
        ("XMLHttpRequest", ["xml", "http", "request"]),
        ("GPTConfig", ["gpt", "config"]),
        ("MLPBlock", ["mlp", "block"]),
        # a bare acronym has no internal boundary and must stay whole
        ("GPT", ["gpt"]),
        # digits are word characters, not separators
        ("print0", ["print0"]),
        ("fp8", ["fp8"]),
    ],
)
def test_split_identifier_words(identifier, expected):
    assert _split_identifier_words(identifier) == expected


def test_acronym_identifier_is_searchable_by_its_parts():
    # The whole point of enrichment: a query for "tokenizer" has to match a
    # symbol named RustBPETokenizer. BM25 matches whole tokens, so the glued
    # form "bpetokenizer" would never hit.
    tags = _identifier_tags("nanochat/tokenizer.py", "RustBPETokenizer", []).split()
    assert "tokenizer" in tags
    assert "bpe" in tags
    assert "rust" in tags


def test_tags_draw_from_path_name_and_decorators():
    tags = _identifier_tags("nanochat/gpt.py", "Trainer.evaluate", ["torch.no_grad()"]).split()
    assert "nanochat" in tags and "gpt" in tags  # path
    assert "trainer" in tags and "evaluate" in tags  # split name
    assert "trainer.evaluate" in tags  # full name kept intact
    assert "torch" in tags and "grad" in tags  # decorator
