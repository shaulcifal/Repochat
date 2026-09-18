"""Convert chunk-level hybrid results into file-level evidence and diversify
across files, before graph expansion (design doc 6.3). Initial chunk results
often over-concentrate in one file; scoring by file and capping chunks per
file preserves coverage across the likely flow."""

from collections import defaultdict

from repochat.domain.models import Chunk

MAX_CHUNKS_PER_FILE_PRE_RERANK = 3


def score_and_diversify(
    ranked_chunks: list[Chunk],
    lexical_only_chunks: list[Chunk],
    *,
    max_files: int = 6,
) -> list[Chunk]:
    """`ranked_chunks` is the fused hybrid result, best first. Returns a flat
    list ordered by (file score, chunk rank within file), capped per file."""
    if not ranked_chunks:
        return []

    position_score = {chunk.id: 1.0 / (i + 1) for i, chunk in enumerate(ranked_chunks)}
    lexical_file_ids = {chunk.file_id for chunk in lexical_only_chunks}

    by_file: dict = defaultdict(list)
    for chunk in ranked_chunks:
        by_file[chunk.file_id].append(chunk)

    file_scores: dict = {}
    for file_id, chunks in by_file.items():
        scores = sorted((position_score[c.id] for c in chunks), reverse=True)
        best = scores[0]
        top3_mean = sum(scores[:3]) / min(3, len(scores))
        lexical_bonus = 1.0 if file_id in lexical_file_ids else 0.0
        file_scores[file_id] = 0.65 * best + 0.25 * top3_mean + 0.10 * lexical_bonus

    ordered_file_ids = sorted(file_scores, key=lambda fid: file_scores[fid], reverse=True)[:max_files]

    result = []
    for file_id in ordered_file_ids:
        chunks_for_file = sorted(by_file[file_id], key=lambda c: position_score[c.id], reverse=True)
        result.extend(chunks_for_file[:MAX_CHUNKS_PER_FILE_PRE_RERANK])
    return result
