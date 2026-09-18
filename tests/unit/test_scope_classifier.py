import pytest

from repochat.generation.llm_provider import LLMProvider, MockLLMProvider
from repochat.retrieval.scope import Scope, policy_for
from repochat.retrieval.scope_classifier import (
    LLMScopeClassifier,
    RuleBasedScopeClassifier,
    _parse_scope,
)


# --------------------------------------------------------------------------
# Rule-based classifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What does save_checkpoint do?", Scope.SYMBOL),
        ("Where is RustBPETokenizer defined?", Scope.SYMBOL),
        ("print0", Scope.SYMBOL),
        ("Trace how a request reaches the database", Scope.REPOSITORY_FLOW),
        ("Walk me through the training pipeline", Scope.REPOSITORY_FLOW),
        ("Explain the data flow end to end", Scope.REPOSITORY_FLOW),
        ("How is authentication configured?", Scope.FEATURE),
        ("How does this project handle errors?", Scope.FEATURE),
    ],
)
def test_rule_based_classification(question, expected):
    assert RuleBasedScopeClassifier().classify(question) == expected


def test_unclassifiable_question_falls_to_feature():
    # No identifier, no flow keyword -> the middle, default setting.
    assert RuleBasedScopeClassifier().classify("Is this project any good?") == Scope.FEATURE


# --------------------------------------------------------------------------
# Label parsing (the model won't reply cleanly every time)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("symbol", Scope.SYMBOL),
        ("  Symbol  ", Scope.SYMBOL),
        ("**repository_flow**", Scope.REPOSITORY_FLOW),
        ("Category: feature", Scope.FEATURE),
        ("`symbol`", Scope.SYMBOL),
        ("repository flow", Scope.REPOSITORY_FLOW),
    ],
)
def test_parse_scope_is_forgiving_about_formatting(raw, expected):
    assert _parse_scope(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "banana", "I am not sure"])
def test_parse_scope_rejects_unrecognizable_output(raw):
    assert _parse_scope(raw) is None


# --------------------------------------------------------------------------
# LLM classifier + fallback behavior
# --------------------------------------------------------------------------


def test_llm_classifier_uses_the_model_answer():
    classifier = LLMScopeClassifier(MockLLMProvider("repository_flow"))
    assert classifier.classify("anything at all") == Scope.REPOSITORY_FLOW


def test_llm_classifier_falls_back_when_output_is_unparseable():
    classifier = LLMScopeClassifier(MockLLMProvider("no idea honestly"))
    # Falls through to the rules, which see a flow keyword.
    assert classifier.classify("Trace the request to the database") == Scope.REPOSITORY_FLOW


def test_llm_classifier_falls_back_when_the_provider_raises():
    class _BrokenProvider(LLMProvider):
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("API is down")

    classifier = LLMScopeClassifier(_BrokenProvider())
    assert classifier.classify("What does save_checkpoint do?") == Scope.SYMBOL


# --------------------------------------------------------------------------
# Policy mapping
# --------------------------------------------------------------------------


def test_narrower_scope_retrieves_less_than_broader_scope():
    symbol = policy_for(Scope.SYMBOL)
    flow = policy_for(Scope.REPOSITORY_FLOW)
    assert symbol.candidate_pool_size < flow.candidate_pool_size
    assert symbol.max_files < flow.max_files
    assert symbol.top_k < flow.top_k
    assert symbol.max_expanded_files < flow.max_expanded_files


def test_every_scope_has_a_policy():
    for scope in Scope:
        assert policy_for(scope).scope == scope
