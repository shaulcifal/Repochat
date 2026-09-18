"""LLMProvider interface -- swapping Groq for another provider later is a
config change, not a rewrite."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class GroqLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int = 1500):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        # gpt-oss models spend tokens on internal reasoning before the
        # visible answer -- too small a budget here comes back empty.
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests -- no network call."""

    def __init__(self, fixed_response: str):
        self._fixed_response = fixed_response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._fixed_response
