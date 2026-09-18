"""EmbeddingProvider interface, so the pipeline can swap implementations
without touching retrieval or chunking code."""

import hashlib
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Runs entirely on the local CPU -- no API call, no rate limit."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, hash-based embedding for tests -- no model download."""

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([byte / 255.0 for byte in digest[: self.dimension]])
        return vectors
