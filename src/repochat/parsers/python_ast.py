"""Extract symbols (modules, classes, functions, methods) from Python source.

Uses the standard-library `ast` module rather than a formatting-normalizing
tool for one reason: chunk citations must point at the *original* file's
line numbers and text, never a reformatted regeneration of it.
"""

import ast
import copy

from repochat.parsers.base import ParsedFile, Symbol


def _signature_of(node: ast.AST) -> str:
    header = copy.copy(node)
    header.body = [ast.Pass()]
    header.decorator_list = []
    return ast.unparse(header).split("\n", 1)[0]


def _source_slice(source_lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(source_lines[start_line - 1 : end_line])


def _function_symbol(node: ast.FunctionDef | ast.AsyncFunctionDef, parent: str | None, source_lines: list[str]) -> Symbol:
    qualified_name = f"{parent}.{node.name}" if parent else node.name
    return Symbol(
        qualified_name=qualified_name,
        kind="method" if parent else "function",
        parent_symbol=parent,
        start_line=node.lineno,
        end_line=node.end_lineno,
        signature=_signature_of(node),
        docstring=ast.get_docstring(node),
        decorators=[ast.unparse(d) for d in node.decorator_list],
        source_text=_source_slice(source_lines, node.lineno, node.end_lineno),
    )


def _class_symbol(node: ast.ClassDef, source_lines: list[str]) -> Symbol:
    return Symbol(
        qualified_name=node.name,
        kind="class",
        parent_symbol=None,
        start_line=node.lineno,
        end_line=node.end_lineno,
        signature=_signature_of(node),
        docstring=ast.get_docstring(node),
        decorators=[ast.unparse(d) for d in node.decorator_list],
        source_text=_source_slice(source_lines, node.lineno, node.end_lineno),
    )


def parse_python_source(source: str) -> ParsedFile:
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ParsedFile(
            language="python",
            parse_status="partial",
            diagnostics=[f"partial parse (syntax error line {exc.lineno}): {exc.msg}"],
        )

    imports: list[str] = []
    symbols: list[Symbol] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_function_symbol(node, parent=None, source_lines=source_lines))
        elif isinstance(node, ast.ClassDef):
            symbols.append(_class_symbol(node, source_lines))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(_function_symbol(child, parent=node.name, source_lines=source_lines))

    return ParsedFile(
        language="python",
        parse_status="ok",
        imports=imports,
        module_docstring=ast.get_docstring(tree),
        symbols=symbols,
    )
