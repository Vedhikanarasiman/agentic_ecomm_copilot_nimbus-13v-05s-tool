from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Anything that turns text into vectors. Swap the implementation,
    not the callers — router_agent.py and rag_service.py only ever
    import this interface, never a specific model or SDK."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class LLMProvider(ABC):
    """Anything that turns messages into a generated response."""

    @abstractmethod
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
        """Returns a dict with at least {"content": str}. If tools were
        passed and the model called one, also includes {"tool_calls": [...]}.
        response_format={"type": "json_object"} requests structured JSON output.
        reasoning_effort ("low"/"medium"/"high") is advisory — only meaningful
        for reasoning models like gpt-oss; providers that don't support it
        should silently ignore it rather than erroring, so callers don't need
        to know which concrete provider is active."""
        ...
