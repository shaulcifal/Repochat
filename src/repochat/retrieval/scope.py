"""Question scope, and the retrieval policy each scope implies.

A question about one named function needs a narrow, precise search. A question
about how data moves across the app needs a wider net and more graph expansion.
Using one fixed policy for both wastes context on the first and starves the
second.
"""

import enum
from dataclasses import dataclass


class Scope(str, enum.Enum):
    SYMBOL = "symbol"  # one specific named function/class/variable
    FEATURE = "feature"  # how some capability or area works
    REPOSITORY_FLOW = "repository_flow"  # how things move across several files
    FOLLOW_UP = "follow_up"  # refers back to a previous answer


@dataclass(frozen=True)
class RetrievalPolicy:
    scope: Scope
    candidate_pool_size: int  # how many hits to pull from each of BM25 / dense
    max_files: int  # how many distinct files survive diversification
    max_expanded_files: int  # graph-expansion breadth (0 disables expansion)
    top_k: int  # how many chunks finally reach the LLM


# Breadth per scope follows the design doc's routing table (6.1).
_POLICIES = {
    Scope.SYMBOL: RetrievalPolicy(
        scope=Scope.SYMBOL, candidate_pool_size=20, max_files=3, max_expanded_files=1, top_k=6
    ),
    Scope.FEATURE: RetrievalPolicy(
        scope=Scope.FEATURE, candidate_pool_size=40, max_files=5, max_expanded_files=3, top_k=8
    ),
    Scope.REPOSITORY_FLOW: RetrievalPolicy(
        scope=Scope.REPOSITORY_FLOW, candidate_pool_size=60, max_files=8, max_expanded_files=5, top_k=12
    ),
    # Follow-ups borrow feature-shaped breadth. The design doc also calls for
    # boosting files cited in the previous answer, which needs the conversation
    # memory that arrives in Phase 6 -- not implemented, so this is breadth only.
    Scope.FOLLOW_UP: RetrievalPolicy(
        scope=Scope.FOLLOW_UP, candidate_pool_size=40, max_files=5, max_expanded_files=3, top_k=8
    ),
}


def policy_for(scope: Scope) -> RetrievalPolicy:
    return _POLICIES[scope]
