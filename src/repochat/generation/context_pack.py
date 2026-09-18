"""Build the labeled evidence pack handed to the answer LLM."""

from dataclasses import dataclass

from repochat.domain.models import Chunk

SYSTEM_PROMPT = (
    "You are RepoChat, answering questions about one specific software repository "
    "using only the SOURCES given below. Cite every factual claim with its source "
    "tag, using plain ASCII square brackets exactly like [S1] or [S2] -- never "
    "full-width, curly, or any other bracket style. If the sources do not contain "
    "enough evidence to answer, say plainly what is missing instead of guessing."
)


@dataclass
class ContextBlock:
    label: str  # "S1", "S2", ...
    chunk: Chunk


def build_context_blocks(chunks: list[Chunk]) -> list[ContextBlock]:
    return [ContextBlock(label=f"S{i + 1}", chunk=chunk) for i, chunk in enumerate(chunks)]


def _symbol_label(chunk: Chunk) -> str:
    return chunk.qualified_name or chunk.file.path


def build_user_prompt(question: str, blocks: list[ContextBlock]) -> str:
    rendered = []
    for block in blocks:
        chunk = block.chunk
        header = f"[{block.label}] {chunk.file.path}:{chunk.start_line}-{chunk.end_line} | {_symbol_label(chunk)}"
        rendered.append(f"{header}\n{chunk.raw_source}")
    sources_text = "\n\n".join(rendered)
    return f"SOURCES\n{sources_text}\n\nQUESTION\n{question}"


def github_permalink(canonical_url: str, commit_sha: str, path: str, start_line: int, end_line: int) -> str:
    return f"{canonical_url}/blob/{commit_sha}/{path}#L{start_line}-L{end_line}"
