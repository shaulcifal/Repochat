from repochat.generation.citation_repair import repair_then_abstain
from repochat.generation.llm_provider import LLMProvider


class _ScriptedProvider(LLMProvider):
    """Returns each response in `responses` in order, one per generate() call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def test_valid_citation_never_triggers_a_repair_call():
    provider = _ScriptedProvider(["should not be used"])
    text = "Training starts in main() [S1]."
    cleaned, confirmed = repair_then_abstain(provider, "sys", "user", text, valid_labels={"S1"})
    assert cleaned == text
    assert confirmed == ["S1"]
    assert provider.calls == []  # no second call needed


def test_invalid_citation_triggers_one_repair_attempt_that_succeeds():
    provider = _ScriptedProvider(["Training starts in main() [S1]."])
    raw_answer = "Training starts in main() [S5]."
    cleaned, confirmed = repair_then_abstain(provider, "sys", "user", raw_answer, valid_labels={"S1"})
    assert len(provider.calls) == 1
    assert cleaned == "Training starts in main() [S1]."
    assert confirmed == ["S1"]


def test_invalid_citation_still_invalid_after_repair_falls_back_to_abstain():
    provider = _ScriptedProvider(["Still using [S9] which does not exist."])
    raw_answer = "Uses a cache [S5]."
    cleaned, confirmed = repair_then_abstain(provider, "sys", "user", raw_answer, valid_labels={"S1"})
    assert confirmed == []
    assert "[S9]" not in cleaned
    assert "S9" in cleaned  # flagged in the trailing note, not silently gone


def test_repair_prompt_mentions_the_invalid_and_valid_labels():
    provider = _ScriptedProvider(["fixed [S1]."])
    repair_then_abstain(provider, "sys", "user prompt", "bad [S9].", valid_labels={"S1", "S2"})
    _, sent_prompt = provider.calls[0]
    assert "S9" in sent_prompt
    assert "S1" in sent_prompt and "S2" in sent_prompt
