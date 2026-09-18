from repochat.chunking.chunker import build_chunk_drafts
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
