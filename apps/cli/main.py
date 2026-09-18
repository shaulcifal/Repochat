import typer

app = typer.Typer(help="RepoChat - repository-aware RAG chatbot")


@app.command()
def index(url: str, ref: str = typer.Option(None, "--ref")):
    """Submit and index a repository."""
    typer.echo(f"Indexing {url} (ref={ref or 'default branch'})... not yet implemented")


@app.command()
def repos():
    """List indexed repositories."""
    typer.echo("No repositories indexed yet.")


@app.command()
def status(repository_id: str):
    """Inspect index progress or failure."""
    typer.echo(f"No status available for {repository_id} yet.")


@app.command()
def chat(repository_id: str):
    """Start or resume revision-scoped chat."""
    typer.echo(f"Chat for {repository_id} not yet implemented.")


@app.command()
def sources(citation_id: str):
    """Inspect a cited source."""
    typer.echo(f"No source found for {citation_id} yet.")


if __name__ == "__main__":
    app()
