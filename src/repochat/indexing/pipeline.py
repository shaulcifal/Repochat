"""Phase 1 indexing: parse every included Python file, chunk it, embed the
chunks, and persist them. Runs between manifest-building and READY."""

from pathlib import Path

from sqlalchemy.orm import Session

from repochat.chunking.chunker import build_chunk_drafts
from repochat.domain.models import (
    Chunk,
    DiagnosticSeverity,
    File,
    FileDiagnostic,
    Revision,
    RevisionState,
)
from repochat.indexing.embedding_provider import EmbeddingProvider
from repochat.parsers.base import ParsedFile
from repochat.parsers.javascript_treesitter import parse_javascript_source
from repochat.parsers.python_ast import parse_python_source

_PARSERS = {
    "python": parse_python_source,
    "javascript": parse_javascript_source,
}


def repo_slug_from_canonical_url(canonical_url: str) -> str:
    return canonical_url.removeprefix("https://github.com/")


def run_indexing(
    session: Session,
    revision: Revision,
    repo_root: Path,
    repo_slug: str,
    embedding_provider: EmbeddingProvider,
) -> None:
    revision.state = RevisionState.PARSING
    session.flush()

    parseable_files = (
        session.query(File)
        .filter(
            File.revision_id == revision.id,
            File.language.in_(list(_PARSERS)),
            File.status.in_(["code", "test"]),
        )
        .all()
    )

    pending_chunks: list[tuple[File, object]] = []  # (File, ChunkDraft)

    for file_row in parseable_files:
        full_source = (repo_root / file_row.path).read_text(encoding="utf-8", errors="replace")
        parse_fn = _PARSERS[file_row.language]
        parsed: ParsedFile = parse_fn(full_source)

        if parsed.parse_status == "partial":
            session.add(
                FileDiagnostic(
                    file_id=file_row.id,
                    revision_id=revision.id,
                    severity=DiagnosticSeverity.WARN,
                    message=f"{file_row.path} - {parsed.diagnostics[0]}, fell back to whole-file chunking",
                    parse_status="partial",
                )
            )

        for draft in build_chunk_drafts(parsed, repo_slug=repo_slug, path=file_row.path, full_source=full_source):
            pending_chunks.append((file_row, draft))

    revision.state = RevisionState.INDEXING
    session.flush()

    if pending_chunks:
        vectors = embedding_provider.embed([draft.embedding_text for _, draft in pending_chunks])
        for (file_row, draft), vector in zip(pending_chunks, vectors):
            session.add(
                Chunk(
                    revision_id=revision.id,
                    file_id=file_row.id,
                    language=draft.language,
                    symbol_kind=draft.symbol_kind,
                    qualified_name=draft.qualified_name,
                    parent_symbol=draft.parent_symbol,
                    start_line=draft.start_line,
                    end_line=draft.end_line,
                    signature=draft.signature,
                    docstring=draft.docstring,
                    raw_source=draft.raw_source,
                    embedding_text=draft.embedding_text,
                    token_count=draft.token_count,
                    content_hash=draft.content_hash,
                    parse_status=draft.parse_status,
                    embedding=vector,
                )
            )

    session.flush()
