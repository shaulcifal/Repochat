from repochat.parsers.python_ast import parse_python_source

SAMPLE = '''"""Module docstring."""
import os
from typing import Optional


class Trainer:
    """Runs the training loop."""

    def train(self, epochs: int) -> None:
        """Run training for the requested epochs."""
        for _ in range(epochs):
            pass


def main():
    trainer = Trainer()
    trainer.train(1)
'''


def test_parses_ok_status():
    parsed = parse_python_source(SAMPLE)
    assert parsed.parse_status == "ok"


def test_extracts_module_docstring():
    parsed = parse_python_source(SAMPLE)
    assert parsed.module_docstring == "Module docstring."


def test_extracts_imports():
    parsed = parse_python_source(SAMPLE)
    assert "import os" in parsed.imports
    assert "from typing import Optional" in parsed.imports


def test_extracts_class_and_method_and_function():
    parsed = parse_python_source(SAMPLE)
    names = {s.qualified_name for s in parsed.symbols}
    assert names == {"Trainer", "Trainer.train", "main"}


def test_method_has_correct_parent_and_kind():
    parsed = parse_python_source(SAMPLE)
    train = next(s for s in parsed.symbols if s.qualified_name == "Trainer.train")
    assert train.kind == "method"
    assert train.parent_symbol == "Trainer"
    assert train.docstring == "Run training for the requested epochs."


def test_function_signature_has_no_body():
    parsed = parse_python_source(SAMPLE)
    train = next(s for s in parsed.symbols if s.qualified_name == "Trainer.train")
    assert train.signature == "def train(self, epochs: int) -> None:"


def test_source_text_matches_original_lines():
    parsed = parse_python_source(SAMPLE)
    main_fn = next(s for s in parsed.symbols if s.qualified_name == "main")
    lines = SAMPLE.splitlines()
    expected = "\n".join(lines[main_fn.start_line - 1 : main_fn.end_line])
    assert main_fn.source_text == expected


def test_syntax_error_yields_partial_status():
    parsed = parse_python_source("def broken(:\n    pass\n")
    assert parsed.parse_status == "partial"
    assert parsed.diagnostics
