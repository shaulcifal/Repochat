from repochat.parsers.javascript_treesitter import parse_javascript_source

SAMPLE = """import { readFile } from 'fs';
const path = require('path');

/**
 * Adds two numbers.
 */
export function add(a, b) {
    return a + b;
}

export class Greeter {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return 'hi ' + this.name;
    }
}

const square = (x) => x * x;
"""


def test_parses_ok_status():
    parsed = parse_javascript_source(SAMPLE)
    assert parsed.parse_status == "ok"


def test_extracts_imports_and_requires():
    parsed = parse_javascript_source(SAMPLE)
    assert any("readFile" in imp for imp in parsed.imports)
    assert any("require('path')" in imp for imp in parsed.imports)


def test_extracts_function_class_method_and_arrow():
    parsed = parse_javascript_source(SAMPLE)
    names = {s.qualified_name for s in parsed.symbols}
    assert names == {"add", "Greeter", "Greeter.constructor", "Greeter.greet", "square"}


def test_exported_function_has_jsdoc_and_correct_kind():
    parsed = parse_javascript_source(SAMPLE)
    add = next(s for s in parsed.symbols if s.qualified_name == "add")
    assert add.kind == "function"
    assert add.docstring == "Adds two numbers."


def test_method_has_correct_parent_and_kind():
    parsed = parse_javascript_source(SAMPLE)
    greet = next(s for s in parsed.symbols if s.qualified_name == "Greeter.greet")
    assert greet.kind == "method"
    assert greet.parent_symbol == "Greeter"


def test_arrow_function_assigned_to_const_is_captured():
    parsed = parse_javascript_source(SAMPLE)
    square = next(s for s in parsed.symbols if s.qualified_name == "square")
    assert square.kind == "function"
    assert "x * x" in square.source_text


def test_signature_excludes_body():
    parsed = parse_javascript_source(SAMPLE)
    add = next(s for s in parsed.symbols if s.qualified_name == "add")
    assert add.signature == "function add(a, b)"


def test_syntax_error_yields_partial_status():
    parsed = parse_javascript_source("function broken( {\n")
    assert parsed.parse_status == "partial"
    assert parsed.diagnostics
