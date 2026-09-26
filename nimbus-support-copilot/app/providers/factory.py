from functools import lru_cache

from app.core.config import get_settings
from app.providers.base import EmbeddingProvider, LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "groq":
        from app.providers.groq_llm import GroqLLM  # lazy: only imported if actually selected

        return GroqLLM(api_key=settings.groq_api_key, model_name=settings.llm_model_name)
    # To add another provider: implement LLMProvider in app/providers/,
    # add a branch here with its own lazy import, flip LLM_PROVIDER in .env.
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "qwen3_local":
        from app.providers.qwen_embeddings import Qwen3LocalEmbedding  # lazy — pulls in torch

        return Qwen3LocalEmbedding(
            model_name=settings.embedding_model_name, dimension=settings.embedding_dim
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
