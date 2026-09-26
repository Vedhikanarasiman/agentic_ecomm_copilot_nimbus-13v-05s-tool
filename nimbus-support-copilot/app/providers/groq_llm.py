import asyncio
from typing import Any

from groq import Groq

from app.providers.base import LLMProvider


class GroqLLM(LLMProvider):
    """Hosted inference of an open-weight model (gpt-oss-20b). Open weights,
    fast hosted inference — resolves the "open source but must handle
    concurrent users" tension without needing our own GPU.

    The groq SDK is sync; we run it in a thread so it doesn't block the
    FastAPI event loop under concurrent requests.
    """

    def __init__(self, api_key: str, model_name: str):
        self._client = Groq(api_key=api_key)
        self._model_name = model_name

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 300,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        def _call():
            kwargs: dict[str, Any] = dict(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response_format:
                kwargs["response_format"] = response_format
            if tools:
                kwargs["tools"] = tools
            if reasoning_effort:
                # gpt-oss-specific: caps how much the model "thinks" before
                # answering. low/medium/high — using "low" on classification
                # and summarization tasks is what actually fixed the
                # token-starvation bug, not just widening max_tokens.
                kwargs["reasoning_effort"] = reasoning_effort
            return self._client.chat.completions.create(**kwargs)

        completion = await asyncio.to_thread(_call)
        choice = completion.choices[0]
        result: dict[str, Any] = {"content": choice.message.content or ""}

        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ]

        # reasoning_tokens is a separate count from completion_tokens for
        # gpt-oss models — this is the number that was silently eating our
        # token budget. Captured explicitly so it's measurable, not guessed.
        reasoning_tokens = None
        details = getattr(completion.usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

        result["usage"] = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        return result
