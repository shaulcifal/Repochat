"""Shared parser-adapter contract: every language parser returns these, so
the chunker and everything downstream never branches on language."""

from dataclasses import dataclass, field


@dataclass
class Symbol:
    qualified_name: str
    kind: str  # "class" | "function" | "method"
    parent_symbol: str | None
    start_line: int
    end_line: int
    signature: str
    docstring: str | None
    decorators: list[str]
    source_text: str


@dataclass
class ParsedFile:
    language: str
    parse_status: str  # "ok" | "partial"
    imports: list[str] = field(default_factory=list)
    module_docstring: str | None = None
    symbols: list[Symbol] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
