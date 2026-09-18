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
    """Runs locally -- GPU if available (sentence-transformers picks this up
    automatically), otherwise CPU. No API call, no rate limit either way."""

    def __init__(self, model_name: str, batch_size: int = 8):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_embedding_dimension()
        # A small default batch, not sentence-transformers' default of 32 --
        # a batch of long code chunks can spike VRAM well past a 4GB card's
        # budget. Backs off further on an actual out-of-memory error so this
        # degrades gracefully across GPUs of different sizes.
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encode_with_backoff(texts, self._batch_size).tolist()

    def _encode_with_backoff(self, texts: list[str], batch_size: int):
        import torch

        try:
            return self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            return self._encode_with_backoff(texts, max(1, batch_size // 2))


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
