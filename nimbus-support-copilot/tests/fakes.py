from app.providers.base import EmbeddingProvider, LLMProvider


class FakeLLMProvider(LLMProvider):
    """Returns pre-scripted responses in order, one per call. Records every
    call it received so tests can assert on what was actually sent — e.g.
    'was reasoning_effort set', 'how many calls happened at all' (that
    second one is how we prove the deterministic order-ID bypass actually
    skips the classification LLM call, not just assume it does)."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(
        self,
        messages,
        *,
        max_tokens=300,
        temperature=0.2,
        response_format=None,
        tools=None,
        reasoning_effort=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "reasoning_effort": reasoning_effort,
            }
        )
        if not self._responses:
            raise AssertionError(
                f"FakeLLMProvider ran out of scripted responses after {len(self.calls)} calls"
            )
        return self._responses.pop(0)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Returns a fixed-size dummy vector — good enough for tests that don't
    care about actual embedding quality, only about what happens after."""

    def __init__(self, dimension: int = 8):
        self._dimension = dimension
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(texts)
        return [[0.1] * self._dimension for _ in texts]

    @property
    def dimension(self):
        return self._dimension
