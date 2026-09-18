"""Phase 3: in-process BM25 lexical search, rebuilt per chat session from
the revision's chunks. Uses true BM25 math via bm25s -- not Postgres's
ts_rank, which is a different ranking function (see the design doc's Fix 2:
an ablation that swaps one for the other isn't measuring what it claims to)."""

import bm25s

from repochat.domain.models import Chunk


class LexicalIndex:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        corpus_texts = [f"{chunk.embedding_text}\n{chunk.tags}" for chunk in chunks]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 20) -> list[Chunk]:
        if not self._chunks:
            return []
        k = min(top_k, len(self._chunks))
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, scores = self._retriever.retrieve(query_tokens, k=k, show_progress=False)
        return [self._chunks[i] for i, score in zip(results[0], scores[0]) if score > 0]


def build_lexical_index(session, revision_id) -> LexicalIndex:
    chunks = session.query(Chunk).filter_by(revision_id=revision_id).all()
    return LexicalIndex(chunks)
