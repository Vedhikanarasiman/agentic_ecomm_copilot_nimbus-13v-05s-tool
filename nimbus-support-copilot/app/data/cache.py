import hashlib
import json

import redis.asyncio as redis

from app.core.config import get_settings

_settings = get_settings()
_redis: redis.Redis = redis.from_url(_settings.redis_url, decode_responses=True)


def _cache_key(query: str) -> str:
    # exact-match only: same question, byte-for-byte after normalization.
    # A semantic cache (similarity-threshold match) would catch paraphrases
    # too, but adds a similarity-search call on every request just to check
    # the cache — deferred until traffic actually justifies it (see
    # trade-offs doc), not built speculatively here.
    normalized = query.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"rag_answer:{digest}"


async def get_cached_answer(query: str) -> dict | None:
    raw = await _redis.get(_cache_key(query))
    return json.loads(raw) if raw else None


async def set_cached_answer(query: str, answer: dict, ttl_seconds: int) -> None:
    await _redis.set(_cache_key(query), json.dumps(answer), ex=ttl_seconds)
