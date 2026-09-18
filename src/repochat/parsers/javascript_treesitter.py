"""Extract symbols from JavaScript/JSX source via Tree-sitter.

Shares the same ParsedFile/Symbol contract as the Python parser (see
parsers/base.py) so the chunker and everything downstream never branches on
language.
"""

from tree_sitter import Language, Node, Parser
import tree_sitter_javascript as tsjs

from repochat.parsers.base import ParsedFile, Symbol

_JS_LANGUAGE = Language(tsjs.language())


def _text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line(node: Node, *, end: bool = False) -> int:
    point = node.end_point if end else node.start_point
    return point[0] + 1  # tree-sitter rows are 0-indexed


def _leading_jsdoc(source: bytes, node: Node) -> str | None:
    prev = node.prev_sibling
    if prev is None or prev.type != "comment":
        return None
    text = _text(source, prev)
    if not text.startswith("/**"):
        return None
    body = text.removeprefix("/**").removesuffix("*/")
    lines = [line.strip().lstrip("*").strip() for line in body.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned or None


def _signature(source: bytes, node: Node, body_node: Node | None) -> str:
    end_byte = body_node.start_byte if body_node is not None else node.end_byte
    return source[node.start_byte : end_byte].decode("utf-8", errors="replace").strip()


def _function_symbol(source: bytes, node: Node, name: str, parent: str | None) -> Symbol:
    body = node.child_by_field_name("body")
    qualified_name = f"{parent}.{name}" if parent else name
    return Symbol(
        qualified_name=qualified_name,
        kind="method" if parent else "function",
        parent_symbol=parent,
        start_line=_line(node),
        end_line=_line(node, end=True),
        signature=_signature(source, node, body),
        docstring=_leading_jsdoc(source, node),
        decorators=[],
        source_text=_text(source, node),
    )


def _class_symbol(source: bytes, node: Node, name: str) -> Symbol:
    body = node.child_by_field_name("body")
    return Symbol(
        qualified_name=name,
        kind="class",
        parent_symbol=None,
        start_line=_line(node),
        end_line=_line(node, end=True),
        signature=_signature(source, node, body),
        docstring=_leading_jsdoc(source, node),
        decorators=[],
        source_text=_text(source, node),
    )


def _is_require_call(node: Node) -> bool:
    if node.type != "call_expression":
        return False
    function_node = node.child_by_field_name("function")
    return function_node is not None and function_node.type == "identifier" and function_node.text == b"require"


def _handle_class(source: bytes, node: Node, symbols: list[Symbol], jsdoc_override: str | None = None) -> None:
    name_node = node.child_by_field_name("name")
    name = _text(source, name_node) if name_node is not None else "(anonymous)"
    class_symbol = _class_symbol(source, node, name)
    if jsdoc_override is not None:
        class_symbol.docstring = jsdoc_override
    symbols.append(class_symbol)

    body = node.child_by_field_name("body")
    if body is None:
        return
    for member in body.children:
        if member.type != "method_definition":
            continue
        member_name_node = member.child_by_field_name("name")
        member_name = _text(source, member_name_node) if member_name_node is not None else "(anonymous)"
        symbols.append(_function_symbol(source, member, member_name, parent=name))


def _handle_lexical_declaration(
    source: bytes, node: Node, symbols: list[Symbol], imports: list[str], jsdoc_override: str | None = None
) -> None:
    declarators = [c for c in node.children if c.type == "variable_declarator"]
    # Only attribute a leading JSDoc block to the function when it's the
    # sole declarator -- `const a = 1, b = () => {}` has no single target.
    single_target_doc = jsdoc_override if len(declarators) == 1 else None

    for declarator in declarators:
        name_node = declarator.child_by_field_name("name")
        value_node = declarator.child_by_field_name("value")
        if name_node is None or value_node is None:
            continue
        if value_node.type in ("arrow_function", "function_expression"):
            symbol = _function_symbol(source, value_node, _text(source, name_node), parent=None)
            if single_target_doc is not None:
                symbol.docstring = single_target_doc
            symbols.append(symbol)
        elif _is_require_call(value_node):
            imports.append(_text(source, node))


def _handle_top_level(
    source: bytes, node: Node, symbols: list[Symbol], imports: list[str], jsdoc_override: str | None = None
) -> None:
    if node.type == "import_statement":
        imports.append(_text(source, node))
        return

    if node.type == "export_statement":
        # A JSDoc block precedes the `export` keyword, not the declaration
        # nested inside it -- look it up here, before unwrapping, and carry
        # it down rather than re-deriving it from the wrong node below.
        doc = jsdoc_override if jsdoc_override is not None else _leading_jsdoc(source, node)
        declaration = node.child_by_field_name("declaration")
        if declaration is not None:
            _handle_top_level(source, declaration, symbols, imports, jsdoc_override=doc)
            return
        if node.child_by_field_name("source") is not None:
            # Re-export, e.g. `export { x } from './y'` -- import-like edge.
            imports.append(_text(source, node))
            return
        for child in node.children:
            if child.type in ("function_declaration", "class_declaration"):
                _handle_top_level(source, child, symbols, imports, jsdoc_override=doc)
                return
        return

    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        name = _text(source, name_node) if name_node is not None else "(anonymous)"
        symbol = _function_symbol(source, node, name, parent=None)
        if jsdoc_override is not None:
            symbol.docstring = jsdoc_override
        symbols.append(symbol)
        return

    if node.type == "class_declaration":
        _handle_class(source, node, symbols, jsdoc_override=jsdoc_override)
        return

    if node.type in ("lexical_declaration", "variable_declaration"):
        _handle_lexical_declaration(source, node, symbols, imports, jsdoc_override=jsdoc_override)
        return


def parse_javascript_source(source_text: str) -> ParsedFile:
    source = source_text.encode("utf-8")
    parser = Parser(_JS_LANGUAGE)
    tree = parser.parse(source)
    root = tree.root_node

    if root.has_error:
        return ParsedFile(
            language="javascript",
            parse_status="partial",
            diagnostics=["partial parse (syntax error detected by Tree-sitter)"],
        )

    imports: list[str] = []
    symbols: list[Symbol] = []
    for node in root.children:
        _handle_top_level(source, node, symbols, imports)

    return ParsedFile(language="javascript", parse_status="ok", imports=imports, symbols=symbols)
