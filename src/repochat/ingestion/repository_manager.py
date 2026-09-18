"""Orchestrates Phase 0: validate -> clone -> classify -> manifest -> READY.

Parsing, chunking, embedding, and lexical indexing are later phases; this
module stops at a revision whose file manifest is fully known and persisted.
"""

import hashlib
import os
import shutil
import stat
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from repochat.domain.models import (
    DiagnosticSeverity,
    File,
    FileDiagnostic,
    FileStatus,
    Repository,
    Revision,
    RevisionState,
)
from repochat.ingestion.clone import CloneError, shallow_clone
from repochat.ingestion.file_policy import MAX_FILES_PER_REVISION, build_manifest
from repochat.ingestion.url_validator import parse_github_url

CLONE_CACHE_ROOT = Path(".clone_cache")


class IndexingError(RuntimeError):
    pass


def _force_remove_readonly(func, path, _exc_info):
    """git marks pack files read-only on Windows; clear that before retrying."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove_clone_dir(dest: Path) -> None:
    shutil.rmtree(dest, onerror=_force_remove_readonly)


def _get_or_create_repository(session: Session, canonical_url: str, default_branch: str) -> Repository:
    repository = session.query(Repository).filter_by(canonical_url=canonical_url).one_or_none()
    if repository is not None:
        return repository
    repository = Repository(canonical_url=canonical_url, default_branch=default_branch)
    session.add(repository)
    session.flush()
    return repository


def index_repository(session: Session, url: str, ref: str | None = None) -> Revision:
    parsed = parse_github_url(url)
    repository = _get_or_create_repository(session, parsed.canonical_url, ref or "HEAD")

    revision = Revision(repository_id=repository.id, state=RevisionState.QUEUED)
    session.add(revision)
    session.flush()

    dest = CLONE_CACHE_ROOT / str(repository.id)
    if dest.exists():
        _remove_clone_dir(dest)

    revision.state = RevisionState.CLONING
    session.flush()

    try:
        commit_sha = shallow_clone(parsed.clone_url, dest, ref)
    except CloneError as exc:
        revision.state = RevisionState.FAILED
        session.commit()
        raise IndexingError(str(exc)) from exc

    revision.commit_sha = commit_sha
    try:
        session.flush()
    except IntegrityError:
        # Same repository + commit already indexed -- reuse it instead of duplicating.
        session.rollback()
        return (
            session.query(Revision)
            .filter_by(repository_id=repository.id, commit_sha=commit_sha)
            .one()
        )

    manifest = build_manifest(dest)

    hasher = hashlib.sha256()
    for entry in sorted(manifest.entries, key=lambda e: e.path):
        hasher.update(entry.path.encode())
        hasher.update((entry.content_hash or "").encode())
    revision.manifest_hash = hasher.hexdigest()

    for entry in manifest.entries:
        file_row = File(
            revision_id=revision.id,
            path=entry.path,
            language=entry.language,
            content_hash=entry.content_hash,
            size_bytes=entry.size_bytes,
            status=FileStatus(entry.status),
        )
        session.add(file_row)
        if entry.exclusion_reason:
            session.flush()  # need file_row.id before referencing it
            session.add(
                FileDiagnostic(
                    file_id=file_row.id,
                    revision_id=revision.id,
                    severity=DiagnosticSeverity.WARN,
                    message=f"{entry.path} - excluded ({entry.exclusion_reason})",
                )
            )

    if manifest.truncated:
        session.add(
            FileDiagnostic(
                file_id=None,
                revision_id=revision.id,
                severity=DiagnosticSeverity.WARN,
                message=f"File-count budget ({MAX_FILES_PER_REVISION}) reached; remaining files were not scanned.",
            )
        )

    # Parsing/chunking/embedding land in later phases; nothing to do yet.
    revision.state = RevisionState.READY
    session.commit()
    return revision
