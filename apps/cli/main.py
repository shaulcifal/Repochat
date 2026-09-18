import os
import sys
from collections import Counter

# Windows' legacy console codepage (cp1252) can't encode every character an
# LLM might produce (smart quotes, non-breaking hyphens, ...); fall back to
# replacing them instead of crashing mid-answer.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from repochat.domain.models import Chunk, Citation, FileDiagnostic, Repository, Revision, RevisionState  # noqa: E402
from repochat.generation.answer_service import answer_question  # noqa: E402
from repochat.generation.context_pack import github_permalink  # noqa: E402
from repochat.generation.llm_provider import GroqLLMProvider  # noqa: E402
from repochat.indexing.embedding_provider import SentenceTransformerEmbeddingProvider  # noqa: E402
from repochat.ingestion.repository_manager import IndexingError, index_repository  # noqa: E402
from repochat.retrieval.lexical_retriever import build_lexical_index  # noqa: E402
from repochat.retrieval.reranker import CrossEncoderRerankerProvider  # noqa: E402
from repochat.storage.db import ensure_schema, get_session  # noqa: E402

app = typer.Typer(help="RepoChat - repository-aware RAG chatbot")
console = Console(width=120)


def _find_repository(session, repository_id: str) -> Repository | None:
    return session.query(Repository).filter_by(public_id=repository_id).one_or_none()


def _latest_revision(session, repository: Repository) -> Revision | None:
    return (
        session.query(Revision)
        .filter_by(repository_id=repository.id)
        .order_by(Revision.created_at.desc())
        .first()
    )


def _embedding_provider():
    console.print("[dim]Loading local embedding model...[/dim]")
    return SentenceTransformerEmbeddingProvider(os.environ["EMBEDDING_MODEL"])


def _llm_provider():
    return GroqLLMProvider(api_key=os.environ["GROQ_API_KEY"], model=os.environ["GROQ_ANSWER_MODEL"])


def _reranker_provider():
    console.print("[dim]Loading reranker model...[/dim]")
    return CrossEncoderRerankerProvider(os.environ["RERANKER_MODEL"])


@app.command()
def index(url: str, ref: str = typer.Option(None, "--ref")):
    """Submit and index a repository."""
    ensure_schema()
    session = get_session()
    try:
        console.print(f"Indexing [bold]{url}[/bold]" + (f" (ref={ref})" if ref else ""))
        embedding_provider = _embedding_provider()
        try:
            revision = index_repository(session, url, ref, embedding_provider=embedding_provider)
        except IndexingError as exc:
            console.print(f"[red]Indexing failed:[/red] {exc}")
            raise typer.Exit(code=1)

        counts = Counter(f.status.value for f in revision.files)
        chunk_count = session.query(Chunk).filter_by(revision_id=revision.id).count()
        console.print(f"At commit [cyan]{revision.commit_sha[:12]}[/cyan]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        for status in ("code", "test", "docs", "config", "excluded"):
            table.add_row(status, str(counts.get(status, 0)))
        console.print(table)

        console.print(f"Indexed: {chunk_count} chunks")
        console.print(f"Status: [green]{revision.state.value}[/green]")
        console.print(f"Repository ID: [bold]{revision.repository.public_id}[/bold]")
    finally:
        session.close()


@app.command()
def repos():
    """List indexed repositories."""
    ensure_schema()
    session = get_session()
    try:
        repositories = session.query(Repository).order_by(Repository.created_at).all()
        if not repositories:
            console.print("No repositories indexed yet.")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("ID")
        table.add_column("URL", overflow="fold")
        table.add_column("Revision")
        table.add_column("State")
        table.add_column("Files", justify="right")
        for repository in repositories:
            revision = _latest_revision(session, repository)
            if revision is None:
                continue
            table.add_row(
                repository.public_id,
                repository.canonical_url,
                revision.commit_sha[:12] or "-",
                revision.state.value,
                str(len(revision.files)),
            )
        console.print(table)
    finally:
        session.close()


@app.command()
def status(repository_id: str):
    """Inspect index progress or failure."""
    ensure_schema()
    session = get_session()
    try:
        repository = _find_repository(session, repository_id)
        if repository is None:
            console.print(f"[red]No repository found with ID {repository_id}[/red]")
            raise typer.Exit(code=1)

        revision = _latest_revision(session, repository)
        if revision is None:
            console.print("No revision recorded for this repository yet.")
            return

        counts = Counter(f.status.value for f in revision.files)
        chunk_count = session.query(Chunk).filter_by(revision_id=revision.id).count()
        console.print(f"State: [bold]{revision.state.value}[/bold]")
        console.print(f"Commit: {revision.commit_sha}")
        console.print(
            "Files: "
            + ", ".join(f"{status} {counts.get(status, 0)}" for status in ("code", "test", "docs", "config", "excluded"))
        )
        console.print(f"Chunks: {chunk_count}")

        diagnostics = (
            session.query(FileDiagnostic).filter_by(revision_id=revision.id).order_by(FileDiagnostic.created_at).all()
        )
        if diagnostics:
            console.print("\nDiagnostics:")
            for diagnostic in diagnostics:
                console.print(f"  [yellow][{diagnostic.severity.value}][/yellow] {diagnostic.message}")
    finally:
        session.close()


def _print_answer(session, repository: Repository, revision: Revision, result) -> None:
    console.print(f"\n[bold]RepoChat >[/bold] {result.answer_markdown}\n")
    if not result.citations:
        return
    console.print("[bold]Sources[/bold]")
    citations_by_label = {c.label: c for c in result.citations}
    for block in result.blocks:
        citation = citations_by_label.get(block.label)
        marker = citation.citation_id if citation else "(unused)"
        chunk = block.chunk
        symbol = chunk.qualified_name or chunk.file.path
        console.print(f"  [{block.label}] {marker} {chunk.file.path}:{chunk.start_line}-{chunk.end_line} {symbol}")


@app.command()
def chat(repository_id: str):
    """Start or resume revision-scoped chat."""
    ensure_schema()
    session = get_session()
    try:
        repository = _find_repository(session, repository_id)
        if repository is None:
            console.print(f"[red]No repository found with ID {repository_id}[/red]")
            raise typer.Exit(code=1)

        revision = _latest_revision(session, repository)
        if revision is None or revision.state != RevisionState.READY:
            console.print("This repository is not indexed and READY yet. Run `repochat index` first.")
            raise typer.Exit(code=1)

        embedding_provider = _embedding_provider()
        llm_provider = _llm_provider()
        model_name = os.environ["GROQ_ANSWER_MODEL"]
        console.print("[dim]Building lexical index...[/dim]")
        lexical_index = build_lexical_index(session, revision.id)
        reranker = _reranker_provider()

        console.print(
            f"RepoChat [{repository.public_id} @ {revision.commit_sha[:7]}] Type /exit to leave.\n"
        )
        while True:
            question = typer.prompt("You")
            if question.strip() in ("/exit", "exit", "quit"):
                break
            result = answer_question(
                session,
                repository_id=repository.id,
                revision=revision,
                question=question,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                llm_model_name=model_name,
                lexical_index=lexical_index,
                reranker=reranker,
            )
            _print_answer(session, repository, revision, result)
    finally:
        session.close()


@app.command()
def sources(citation_id: str):
    """Inspect a cited source."""
    ensure_schema()
    session = get_session()
    try:
        citation = session.query(Citation).filter_by(citation_id=citation_id).one_or_none()
        if citation is None:
            console.print(f"[red]No source found for {citation_id}[/red]")
            raise typer.Exit(code=1)

        chunk = citation.chunk
        repository = chunk.revision.repository
        symbol = chunk.qualified_name or chunk.file.path

        console.print(f"{chunk.file.path}:{chunk.start_line}-{chunk.end_line} | {symbol}\n")
        console.print(chunk.raw_source)
        console.print(
            "\n"
            + github_permalink(repository.canonical_url, citation.revision_sha, chunk.file.path, chunk.start_line, chunk.end_line)
        )
    finally:
        session.close()


if __name__ == "__main__":
    app()
