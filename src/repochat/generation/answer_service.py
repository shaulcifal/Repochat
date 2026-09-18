"""Orchestrates one question: retrieve -> build context -> generate ->
repair-then-abstain citation validation -> persist. Conversation memory /
follow-up resolution is a Phase 6 concern; each call here is independent."""

import json
import time
import uuid

from sqlalchemy.orm import Session

from repochat.domain.models import AnswerEvent, Citation, Revision, RetrievalTrace
from repochat.generation.citation_repair import repair_then_abstain
from repochat.generation.context_pack import SYSTEM_PROMPT, ContextBlock, build_context_blocks, build_user_prompt
from repochat.generation.llm_provider import LLMProvider
from repochat.indexing.embedding_provider import EmbeddingProvider
from repochat.retrieval.reranker import RerankerProvider
from repochat.retrieval.traced_pipeline import run_retrieval


class AnswerResult:
    def __init__(self, answer_markdown: str, blocks: list[ContextBlock], citations: list[Citation]):
        self.answer_markdown = answer_markdown
        self.blocks = blocks
        self.citations = citations


def _persist_no_evidence_answer(session, *, repository_id, revision, question, model_name) -> AnswerResult:
    answer_markdown = "I don't have any indexed evidence for this repository yet."
    answer_event = AnswerEvent(
        repository_id=repository_id,
        revision_id=revision.id,
        question=question,
        answer_markdown=answer_markdown,
        model=model_name,
        latency_ms=0,
    )
    session.add(answer_event)
    session.commit()
    return AnswerResult(answer_markdown, [], [])


def answer_question(
    session: Session,
    *,
    repository_id: uuid.UUID,
    revision: Revision,
    question: str,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    llm_model_name: str,
    lexical_index,
    reranker: RerankerProvider,
    top_k: int = 8,
) -> AnswerResult:
    start = time.monotonic()

    [query_vector] = embedding_provider.embed([question])
    retrieval = run_retrieval(
        session,
        revision.id,
        query=question,
        query_vector=query_vector,
        lexical_index=lexical_index,
        reranker=reranker,
        top_k=top_k,
    )
    chunks = retrieval.chunks

    if not chunks:
        return _persist_no_evidence_answer(
            session, repository_id=repository_id, revision=revision, question=question, model_name=llm_model_name
        )

    blocks = build_context_blocks(chunks)
    user_prompt = build_user_prompt(question, blocks)
    raw_answer = llm_provider.generate(SYSTEM_PROMPT, user_prompt)

    valid_labels = {block.label for block in blocks}
    cleaned_answer, cited_labels = repair_then_abstain(
        llm_provider, SYSTEM_PROMPT, user_prompt, raw_answer, valid_labels
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    answer_event = AnswerEvent(
        repository_id=repository_id,
        revision_id=revision.id,
        question=question,
        answer_markdown=cleaned_answer,
        model=llm_model_name,
        latency_ms=latency_ms,
    )
    session.add(answer_event)
    session.flush()

    trace = RetrievalTrace(
        answer_event_id=answer_event.id,
        lexical_results=json.dumps(retrieval.trace["lexical_results"]),
        dense_results=json.dumps(retrieval.trace["dense_results"]),
        rrf_results=json.dumps(retrieval.trace["rrf_results"]),
        graph_expansion=json.dumps(retrieval.trace["graph_expansion"]),
        reranker_results=json.dumps(retrieval.trace["reranker_results"]),
        final_chunks=json.dumps(retrieval.trace["final_chunks"]),
        context_chunks=json.dumps(retrieval.trace["context_chunks"]),
    )
    session.add(trace)
    session.flush()

    label_to_block = {block.label: block for block in blocks}
    citations = []
    for label in cited_labels:
        block = label_to_block[label]
        citation = Citation(
            citation_id=f"{answer_event.public_id}-{label}",
            answer_event_id=answer_event.id,
            label=label,
            chunk_id=block.chunk.id,
            revision_sha=revision.commit_sha,
        )
        session.add(citation)
        citations.append(citation)

    session.commit()
    return AnswerResult(cleaned_answer, blocks, citations)
