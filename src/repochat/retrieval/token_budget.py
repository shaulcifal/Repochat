"""Fit the ranked chunk list into a fixed context token budget by dropping
low-ranked chunks first -- never truncating through a source block or
removing its citation header (design doc 7.2)."""

DEFAULT_CONTEXT_TOKEN_BUDGET = 6_000


def fit_to_budget(ranked_chunks: list, budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET) -> list:
    selected = []
    used = 0
    for chunk in ranked_chunks:
        if used + chunk.token_count > budget and selected:
            break
        selected.append(chunk)
        used += chunk.token_count
    return selected
