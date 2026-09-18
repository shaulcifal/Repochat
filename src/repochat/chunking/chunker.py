"""Turn a parsed file into chunk drafts: one per symbol, plus an optional
whole-module overview chunk. Embedding text follows the recipe in the design
doc so identifier queries and behavioral queries can both match."""

import hashlib
from dataclasses import dataclass

from repochat.parsers.python_ast import ParsedFile

MAX_MODULE_CHUNK_CHARS = 4_000


@dataclass
class ChunkDraft:
    language: str
    symbol_kind: str
    qualified_name: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    signature: str | None
    docstring: str | None
    raw_source: str
    embedding_text: str
    token_count: int
    content_hash: str
    parse_status: str


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _build_embedding_text(
    *,
    repo_slug: str,
    path: str,
    language: str,
    symbol_kind: str,
    qualified_name: str | None,
    start_line: int,
    end_line: int,
    parent_label: str,
    signature: str | None,
    docstring: str | None,
    raw_source: str,
) -> str:
    lines = [
        f"[repository: {repo_slug}] [path: {path}] [language: {language}]",
        f"[symbol: {qualified_name or path}] [kind: {symbol_kind}] [lines: {start_line}-{end_line}]",
        f"[parent: {parent_label}]" + (f" [signature: {signature}]" if signature else ""),
    ]
    if docstring:
        short_doc = docstring.strip().splitlines()[0][:300]
        lines.append(f"[docstring: {short_doc}]")
    lines.append("SOURCE")
    lines.append(raw_source)
    return "\n".join(lines)


def _module_draft(parsed: ParsedFile, *, repo_slug: str, path: str, full_source: str, parse_status: str) -> ChunkDraft:
    end_line = max(1, len(full_source.splitlines()))
    docstring = parsed.module_docstring if parse_status == "ok" else None
    embedding_text = _build_embedding_text(
        repo_slug=repo_slug,
        path=path,
        language=parsed.language,
        symbol_kind="module",
        qualified_name=None,
        start_line=1,
        end_line=end_line,
        parent_label="module",
        signature=None,
        docstring=docstring,
        raw_source=full_source,
    )
    return ChunkDraft(
        language=parsed.language,
        symbol_kind="module",
        qualified_name=None,
        parent_symbol=None,
        start_line=1,
        end_line=end_line,
        signature=None,
        docstring=docstring,
        raw_source=full_source,
        embedding_text=embedding_text,
        token_count=_approx_token_count(embedding_text),
        content_hash=hashlib.sha256(full_source.encode()).hexdigest(),
        parse_status=parse_status,
    )


def build_chunk_drafts(parsed: ParsedFile, *, repo_slug: str, path: str, full_source: str) -> list[ChunkDraft]:
    if parsed.parse_status == "partial":
        # Syntax error: fall back to one whole-file chunk, lower confidence,
        # rather than losing the file from retrieval entirely.
        return [_module_draft(parsed, repo_slug=repo_slug, path=path, full_source=full_source, parse_status="partial")]

    drafts: list[ChunkDraft] = []

    if (parsed.module_docstring or parsed.imports) and len(full_source) <= MAX_MODULE_CHUNK_CHARS:
        drafts.append(_module_draft(parsed, repo_slug=repo_slug, path=path, full_source=full_source, parse_status="ok"))

    for symbol in parsed.symbols:
        parent_label = f"class {symbol.parent_symbol}" if symbol.parent_symbol else "module"
        embedding_text = _build_embedding_text(
            repo_slug=repo_slug,
            path=path,
            language=parsed.language,
            symbol_kind=symbol.kind,
            qualified_name=symbol.qualified_name,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            parent_label=parent_label,
            signature=symbol.signature,
            docstring=symbol.docstring,
            raw_source=symbol.source_text,
        )
        drafts.append(
            ChunkDraft(
                language=parsed.language,
                symbol_kind=symbol.kind,
                qualified_name=symbol.qualified_name,
                parent_symbol=symbol.parent_symbol,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                signature=symbol.signature,
                docstring=symbol.docstring,
                raw_source=symbol.source_text,
                embedding_text=embedding_text,
                token_count=_approx_token_count(embedding_text),
                content_hash=hashlib.sha256(symbol.source_text.encode()).hexdigest(),
                parse_status="ok",
            )
        )

    return drafts
