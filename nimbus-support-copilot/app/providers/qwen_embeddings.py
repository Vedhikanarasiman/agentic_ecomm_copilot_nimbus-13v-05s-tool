import asyncio

from sentence_transformers import SentenceTransformer

from app.providers.base import EmbeddingProvider


class Qwen3LocalEmbedding(EmbeddingProvider):
    """Runs Qwen3-Embedding-0.6B locally via sentence-transformers. Small
    enough (0.6B) to stay off the concurrency-bottleneck path on CPU as
    long as calls are batched — batching is the caller's job (pass all
    chunks for a document in one .embed() call, not one chunk at a time).

    Loaded once per process (model load is slow; embedding calls are not).
    """

    def __init__(self, model_name: str, dimension: int = 1024):
        # truncate_dim uses the model's native Matryoshka support — this
        # is a config choice, not a limitation of the model.
        self._model = SentenceTransformer(model_name, truncate_dim=dimension)
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def _encode():
            return self._model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        return await asyncio.to_thread(_encode)

    @property
    def dimension(self) -> int:
        return self._dimension
