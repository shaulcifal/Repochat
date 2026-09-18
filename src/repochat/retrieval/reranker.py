"""RerankerProvider interface: score (question, candidate) pairs together --
more precise than dense retrieval's precomputed embeddings, but too slow to
run over the whole repository, so it only ever sees the small pool hybrid
retrieval + graph expansion already narrowed down (design doc 6.5).

Using cross-encoder/ms-marco-MiniLM-L-6-v2 here rather than the doc's
primary recommendation (a 4B-parameter code-specific reranker) -- that
model is an ~8GB download with CPU-only inference on this machine, so this
generic baseline (the doc's own documented ablation-comparison option) is
promoted to primary given real hardware constraints, same reasoning as the
Phase 1 embedding-model swap.
"""

from abc import ABC, abstractmethod


class RerankerProvider(ABC):
    @abstractmethod
    def score(self, question: str, candidate_texts: list[str]) -> list[float]:
        ...


class CrossEncoderRerankerProvider(RerankerProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, question: str, candidate_texts: list[str]) -> list[float]:
        if not candidate_texts:
            return []
        pairs = [(question, text) for text in candidate_texts]
        return [float(s) for s in self._model.predict(pairs)]


class MockRerankerProvider(RerankerProvider):
    """Deterministic provider for tests -- no model download."""

    def __init__(self, fixed_scores: list[float] | None = None):
        self._fixed_scores = fixed_scores

    def score(self, question: str, candidate_texts: list[str]) -> list[float]:
        if self._fixed_scores is not None:
            return self._fixed_scores
        return [0.0] * len(candidate_texts)
