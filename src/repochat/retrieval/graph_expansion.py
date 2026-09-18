"""Controlled 1-hop import-graph expansion: add directly-imported files'
most relevant chunks as supporting candidates.

Expansion is a candidate-generation step, not proof that every neighbor is
relevant (design doc 6.4) -- breadth is capped rather than adding whole
files blindly.
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from repochat.domain.models import Chunk, GraphEdge

MAX_SEED_FILES_EXPANDED = 3
MAX_CHUNKS_PER_EXPANDED_FILE = 2
MAX_EXPANDED_FILES = 4


def expand_via_imports(
    session: Session,
    revision_id,
    seed_chunks: list[Chunk],
    query_vector: list[float],
) -> list[Chunk]:
    seed_file_ids: list = []
    seen = set()
    for chunk in seed_chunks:
        if chunk.file_id not in seen:
            seen.add(chunk.file_id)
            seed_file_ids.append(chunk.file_id)
    seed_file_ids = seed_file_ids[:MAX_SEED_FILES_EXPANDED]
    if not seed_file_ids:
        return []

    # Follow both directions: what a seed file imports, and what imports the
    # seed file. Seeding on a widely-used utility module (the callee) is common
    # -- without the incoming direction, the higher-level file that actually
    # asked the question about (the caller) would never be reachable.
    edges = (
        session.query(GraphEdge)
        .filter(
            GraphEdge.revision_id == revision_id,
            (GraphEdge.source_file_id.in_(seed_file_ids)) | (GraphEdge.target_file_id.in_(seed_file_ids)),
        )
        .all()
    )

    target_file_ids: list = []
    for edge in edges:
        if edge.source_file_id in seed_file_ids:
            neighbor = edge.target_file_id
        else:
            neighbor = edge.source_file_id
        if neighbor in seen or neighbor in target_file_ids:
            continue
        target_file_ids.append(neighbor)
    if not target_file_ids:
        return []

    # A widely-imported hub file can have dozens of neighbors; picking the
    # first N in arbitrary row order would let that ordering, not relevance,
    # decide which neighbor actually reaches the answer. Rank candidate
    # neighbor files by their single best-matching chunk instead.
    best_distance_stmt = (
        select(Chunk.file_id, func.min(Chunk.embedding.cosine_distance(query_vector)).label("best_distance"))
        .where(Chunk.revision_id == revision_id, Chunk.file_id.in_(target_file_ids))
        .group_by(Chunk.file_id)
    )
    best_distance_by_file = {row.file_id: row.best_distance for row in session.execute(best_distance_stmt)}
    target_file_ids.sort(key=lambda file_id: best_distance_by_file.get(file_id, float("inf")))
    target_file_ids = target_file_ids[:MAX_EXPANDED_FILES]

    added: list[Chunk] = []
    for file_id in target_file_ids:
        stmt = (
            select(Chunk)
            .where(Chunk.revision_id == revision_id, Chunk.file_id == file_id)
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(MAX_CHUNKS_PER_EXPANDED_FILE)
        )
        added.extend(session.scalars(stmt))
    return added
