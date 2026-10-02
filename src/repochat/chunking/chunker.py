"""Turn a parsed file into chunk drafts: one per symbol, plus an optional
whole-module overview chunk. Embedding text follows the recipe in the design
doc so identifier queries and behavioral queries can both match."""

import hashlib
import re
from dataclasses import dataclass

from repochat.parsers.base import ParsedFile

MAX_MODULE_CHUNK_CHARS = 4_000

# Two zero-width boundaries, because one isn't enough:
#   (?<=[a-z0-9])(?=[A-Z])      camelCase      -> camel | Case
#   (?<=[A-Z])(?=[A-Z][a-z])    an acronym run -> BPE | Tokenizer
# Without the second, every character inside an acronym is uppercase, so no
# boundary fires and "RustBPETokenizer" stays glued as "bpetokenizer" -- which
# BM25 will never match against a query for "tokenizer". That matters most in
# exactly the identifiers where acronyms cluster: GPT, BPE, KV, MLP, FP8.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^A-Za-z0-9]+")


def _split_identifier_words(name: str) -> list[str]:
    """"save_checkpoint" -> ["save", "checkpoint"]; "RustBPETokenizer" ->
    ["rust", "bpe", "tokenizer"]. Lets an exact identifier match lexically
    even split across snake_case/camelCase/acronym/path boundaries."""
    words = []
    for part in _NON_WORD.split(name):
        if not part:
            continue
        words.extend(sub.lower() for sub in _CAMEL_BOUNDARY.split(part) if sub)
    return words


def _identifier_tags(path: str, qualified_name: str | None, decorators: list[str]) -> str:
    tokens: set[str] = set()
    for segment in path.split("/"):
        tokens.update(_split_identifier_words(segment))
    if qualified_name:
        tokens.add(qualified_name.lower())
        for part in qualified_name.split("."):
            tokens.update(_split_identifier_words(part))
    for decorator in decorators:
        tokens.update(_split_identifier_words(decorator))
    return " ".join(sorted(tokens))


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
    tags: str
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
        tags=_identifier_tags(path, None, []),
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
                tags=_identifier_tags(path, symbol.qualified_name, symbol.decorators),
                token_count=_approx_token_count(embedding_text),
                content_hash=hashlib.sha256(symbol.source_text.encode()).hexdigest(),
                parse_status="ok",
            )
        )

    return drafts
