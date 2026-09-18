"""Core entities: repository, revision, file, and file-level diagnostics."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def _new_public_id() -> str:
    return f"repo_{uuid.uuid4().hex[:8]}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RevisionState(str, enum.Enum):
    QUEUED = "QUEUED"
    CLONING = "CLONING"
    PARSING = "PARSING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"
    DELETED = "DELETED"


class FileStatus(str, enum.Enum):
    CODE = "code"
    DOCS = "docs"
    CONFIG = "config"
    TEST = "test"
    EXCLUDED = "excluded"


class DiagnosticSeverity(str, enum.Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Repository(Base):
    __tablename__ = "repository"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    public_id: Mapped[str] = mapped_column(String, unique=True, default=_new_public_id)
    canonical_url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    default_branch: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    revisions: Mapped[list["Revision"]] = relationship(back_populates="repository")


class Revision(Base):
    __tablename__ = "revision"
    __table_args__ = (UniqueConstraint("repository_id", "commit_sha", name="uq_revision_repo_commit"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    repository_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repository.id"), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False, default="")
    state: Mapped[RevisionState] = mapped_column(
        SqlEnum(RevisionState, native_enum=False, validate_strings=True), default=RevisionState.QUEUED
    )
    manifest_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    repository: Mapped["Repository"] = relationship(back_populates="revisions")
    files: Mapped[list["File"]] = relationship(back_populates="revision")
    diagnostics: Mapped[list["FileDiagnostic"]] = relationship(back_populates="revision")


class File(Base):
    __tablename__ = "file"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("revision.id"), nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[FileStatus] = mapped_column(SqlEnum(FileStatus, native_enum=False, validate_strings=True))

    revision: Mapped["Revision"] = relationship(back_populates="files")
    diagnostics: Mapped[list["FileDiagnostic"]] = relationship(back_populates="file")


class FileDiagnostic(Base):
    """Every syntax failure, partial parse, or skipped-file reason, kept past ingestion."""

    __tablename__ = "file_diagnostic"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    file_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("file.id"), nullable=True)
    revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("revision.id"), nullable=False)
    severity: Mapped[DiagnosticSeverity] = mapped_column(
        SqlEnum(DiagnosticSeverity, native_enum=False, validate_strings=True)
    )
    message: Mapped[str] = mapped_column(String, nullable=False)
    parse_status: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    file: Mapped["File | None"] = relationship(back_populates="diagnostics")
    revision: Mapped["Revision"] = relationship(back_populates="diagnostics")
