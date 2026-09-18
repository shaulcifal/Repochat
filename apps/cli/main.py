from collections import Counter

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from repochat.domain.models import FileDiagnostic, Repository, Revision  # noqa: E402
from repochat.ingestion.repository_manager import IndexingError, index_repository  # noqa: E402
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


@app.command()
def index(url: str, ref: str = typer.Option(None, "--ref")):
    """Submit and index a repository."""
    ensure_schema()
    session = get_session()
    try:
        console.print(f"Indexing [bold]{url}[/bold]" + (f" (ref={ref})" if ref else ""))
        try:
            revision = index_repository(session, url, ref)
        except IndexingError as exc:
            console.print(f"[red]Indexing failed:[/red] {exc}")
            raise typer.Exit(code=1)

        counts = Counter(f.status.value for f in revision.files)
        console.print(f"At commit [cyan]{revision.commit_sha[:12]}[/cyan]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        for status in ("code", "test", "docs", "config", "excluded"):
            table.add_row(status, str(counts.get(status, 0)))
        console.print(table)

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
        console.print(f"State: [bold]{revision.state.value}[/bold]")
        console.print(f"Commit: {revision.commit_sha}")
        console.print(
            "Files: "
            + ", ".join(f"{status} {counts.get(status, 0)}" for status in ("code", "test", "docs", "config", "excluded"))
        )

        diagnostics = (
            session.query(FileDiagnostic).filter_by(revision_id=revision.id).order_by(FileDiagnostic.created_at).all()
        )
        if diagnostics:
            console.print("\nDiagnostics:")
            for diagnostic in diagnostics:
                console.print(f"  [yellow][{diagnostic.severity.value}][/yellow] {diagnostic.message}")
    finally:
        session.close()


@app.command()
def chat(repository_id: str):
    """Start or resume revision-scoped chat."""
    typer.echo(f"Chat for {repository_id} not yet implemented (Phase 6).")


@app.command()
def sources(citation_id: str):
    """Inspect a cited source."""
    typer.echo(f"No source found for {citation_id} yet (Phase 5).")


if __name__ == "__main__":
    app()
