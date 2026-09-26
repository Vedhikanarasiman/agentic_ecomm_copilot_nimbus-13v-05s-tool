from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database and cache
    database_url: str
    redis_url: str

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Provider selection — the entire point of app/providers/factory.py
    llm_provider: str = "groq"          # "groq" | (future) "local_llama_cpp" | "openai"
    embedding_provider: str = "qwen3_local"  # "qwen3_local" | (future) "hf_inference"

    groq_api_key: str = ""
    llm_model_name: str = "openai/gpt-oss-20b"

    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024

    # Retrieval
    rag_top_k: int = 4
    rag_similarity_threshold: float = 0.55

    # Response bounds — bounded on purpose, see design notes in README
    rag_max_tokens: int = 300
    tool_call_max_tokens: int = 200  # was 100 — same gpt-oss reasoning-token starvation as router_max_tokens
    router_max_tokens: int = 200  # gpt-oss reasons before emitting JSON — 50 was too tight, caused 400s

    cache_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    """Cached so Settings() is only parsed once per process, not per request."""
    return Settings()
