"""Classify a question's scope so retrieval can size itself accordingly.

The classifier returns a label and nothing else -- it never answers the
repository question, and it never decides which chunks are relevant. It only
picks how wide to cast the net.

It runs on a small, cheap model kept separate from the answer model, so routing
stays inexpensive and independently replaceable (design doc 6.1). When that
model is unavailable or says something unrecognizable, a deterministic
rule-based classifier takes over rather than failing the request.
"""

import re
from abc import ABC, abstractmethod

from repochat.generation.llm_provider import LLMProvider
from repochat.retrieval.scope import Scope

_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify a question about a software repository into exactly one category. "
    "Reply with only the category name and nothing else.\n\n"
    "Categories:\n"
    "symbol - asks about one specific named function, class, method, or variable\n"
    "feature - asks how some capability or area of the codebase works\n"
    "repository_flow - asks how data or control moves across several files, or to "
    "trace a path end to end"
)

_FLOW_KEYWORDS = (
    "trace",
    "end to end",
    "end-to-end",
    "walk me through",
    "pipeline",
    "flow",
    "lifecycle",
    "life cycle",
    "start to finish",
    "from request to",
    "across files",
)

_SYMBOL_KEYWORDS = (
    "where is",
    "what does",
    "defined",
    "signature",
    "return",
)

# snake_case, camelCase/PascalCase, digit-suffixed (print0), dotted paths, or
# an explicit call like foo(). Deliberately does NOT try to catch plain
# lowercase words -- "main" and "norm" are real function names, but so are
# ordinary English words, and guessing wrong here mis-sizes the search. Those
# ambiguous cases are what the model-based classifier is for; these rules only
# have to be sane, not clever.
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:[a-z0-9]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z]*|[A-Z][a-z]+[A-Z][A-Za-z]*"
    r"|[a-z]+\d+|\w+\.\w+|\w+\(\))\b"
)


class ScopeClassifier(ABC):
    @abstractmethod
    def classify(self, question: str) -> Scope:
        ...


class RuleBasedScopeClassifier(ScopeClassifier):
    """Deterministic, no network. Used as the fallback, and usable alone."""

    def classify(self, question: str) -> Scope:
        text = question.strip()
        lowered = text.lower()

        if any(keyword in lowered for keyword in _FLOW_KEYWORDS):
            return Scope.REPOSITORY_FLOW

        if _IDENTIFIER_PATTERN.search(text) and any(k in lowered for k in _SYMBOL_KEYWORDS):
            return Scope.SYMBOL

        # A bare identifier with almost no surrounding prose is a symbol lookup.
        if _IDENTIFIER_PATTERN.search(text) and len(text.split()) <= 4:
            return Scope.SYMBOL

        return Scope.FEATURE


class LLMScopeClassifier(ScopeClassifier):
    """Asks a small model for the label, with a rule-based safety net.

    Works with any LLMProvider, so it can be driven by a cheap model in
    production and by a mock in tests.
    """

    def __init__(self, llm_provider: LLMProvider, fallback: ScopeClassifier | None = None):
        self._llm = llm_provider
        self._fallback = fallback or RuleBasedScopeClassifier()

    def classify(self, question: str) -> Scope:
        try:
            raw = self._llm.generate(_CLASSIFIER_SYSTEM_PROMPT, question)
        except Exception:
            # A routing decision is never worth failing the whole question over.
            return self._fallback.classify(question)

        scope = _parse_scope(raw)
        return scope if scope is not None else self._fallback.classify(question)


class MockScopeClassifier(ScopeClassifier):
    """Returns a fixed scope. For tests."""

    def __init__(self, scope: Scope = Scope.FEATURE):
        self._scope = scope

    def classify(self, question: str) -> Scope:
        return self._scope


def _parse_scope(raw: str) -> Scope | None:
    """Forgiving label extraction.

    A model told to reply with one word will still sometimes wrap it in
    markdown, prefix it with 'Category:', or write 'follow up' with a space.
    Same lesson as citation parsing: match the content, not the formatting.
    """
    if not raw:
        return None
    normalized = re.sub(r"[^a-z]+", "_", raw.strip().lower()).strip("_")
    for scope in Scope:
        if scope.value in normalized:
            return scope
    return None
