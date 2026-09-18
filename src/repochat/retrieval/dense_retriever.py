"""Phase 1: dense-only retrieval. Hybrid (BM25 + RRF) arrives in Phase 3."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from repochat.domain.models import Chunk


def search_dense(session: Session, revision_id: uuid.UUID, query_vector: list[float], top_k: int = 8) -> list[Chunk]:
    # Every query is scoped to one revision -- chunks from other repositories
    # or older revisions of the same repository can never leak into results.
    stmt = (
        select(Chunk)
        .where(Chunk.revision_id == revision_id)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    return list(session.scalars(stmt))
