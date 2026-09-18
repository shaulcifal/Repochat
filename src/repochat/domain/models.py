"""Core entities: repository, revision, file, and file-level diagnostics."""

import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum as SqlEnum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSION = 768  # Alibaba-NLP/gte-modernbert-base


class Base(DeclarativeBase):
    pass


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def _new_public_id() -> str:
    return f"repo_{uuid.uuid4().hex[:8]}"


def _new_answer_event_public_id() -> str:
    return f"ae_{uuid.uuid4().hex[:8]}"


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


class Chunk(Base):
    """A single retrieval unit: one symbol (module/class/function/method)."""

    __tablename__ = "chunk"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("revision.id"), nullable=False)
    file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("file.id"), nullable=False)

    language: Mapped[str] = mapped_column(String, nullable=False)
    symbol_kind: Mapped[str] = mapped_column(String, nullable=False)  # module|class|function|method
    qualified_name: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_symbol: Mapped[str | None] = mapped_column(String, nullable=True)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    signature: Mapped[str | None] = mapped_column(String, nullable=True)
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str] = mapped_column(Text, nullable=False, default="")  # space-joined identifier-enrichment tokens

    raw_source: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    parse_status: Mapped[str] = mapped_column(String, nullable=False)  # ok|partial

    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    revision: Mapped["Revision"] = relationship()
    file: Mapped["File"] = relationship()


class GraphEdge(Base):
    """A file-to-file relationship discovered from imports. A retrieval aid
    and candidate-generation signal, not proof of what actually executes."""

    __tablename__ = "graph_edge"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("revision.id"), nullable=False)
    source_file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("file.id"), nullable=False)
    target_file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("file.id"), nullable=False)
    edge_type: Mapped[str] = mapped_column(String, nullable=False)  # "imports"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnswerEvent(Base):
    __tablename__ = "answer_event"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    public_id: Mapped[str] = mapped_column(String, unique=True, default=_new_answer_event_public_id)
    repository_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repository.id"), nullable=False)
    revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("revision.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    citations: Mapped[list["Citation"]] = relationship(back_populates="answer_event")
    # RetrievalTrace points back at this row via answer_event_id -- a single
    # FK direction, not two (a two-way FK pair is an unresolvable DROP-order
    # cycle in Postgres/SQLAlchemy unless the constraints are explicitly
    # named; simplest is to just not create the cycle).
    retrieval_trace: Mapped["RetrievalTrace | None"] = relationship(back_populates="answer_event", uselist=False)


class RetrievalTrace(Base):
    """One row per answer, holding every intermediate retrieval stage's
    output -- what makes a bad answer debuggable stage-by-stage instead of
    only having the final answer to go on (design doc Fix 6)."""

    __tablename__ = "retrieval_trace"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    answer_event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("answer_event.id"), nullable=False)
    scope: Mapped[str | None] = mapped_column(String, nullable=True)  # scope classifier not built yet (Phase 6)
    lexical_results: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    dense_results: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    rrf_results: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    graph_expansion: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    reranker_results: Mapped[str] = mapped_column(Text, nullable=False)  # JSON, raw + normalized
    final_chunks: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    context_chunks: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    answer_event: Mapped["AnswerEvent"] = relationship(back_populates="retrieval_trace")


class Citation(Base):
    """A durable, globally-addressable citation: '<answer_event_id>-S<label>'."""

    __tablename__ = "citation"

    citation_id: Mapped[str] = mapped_column(String, primary_key=True)
    answer_event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("answer_event.id"), nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)  # "S1", "S2", ...
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chunk.id"), nullable=False)
    revision_sha: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    answer_event: Mapped["AnswerEvent"] = relationship(back_populates="citations")
    chunk: Mapped["Chunk"] = relationship()
