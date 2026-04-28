from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Protocol

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - exercised implicitly by environments without redis installed
    Redis = None

    class RedisError(Exception):
        pass


@dataclass
class IngestionProgress:
    signature: str
    total_blocks: int
    completed_blocks: int
    thread_id: str | None = None


class IngestionStoreBackend(Protocol):
    def get(self, conversation_key: str) -> IngestionProgress | None: ...

    def set(self, conversation_key: str, progress: IngestionProgress, ttl_seconds: int) -> None: ...

    def delete(self, conversation_key: str) -> None: ...

    def clear(self) -> None: ...


class InMemoryIngestionStore:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[IngestionProgress, float]] = {}

    def get(self, conversation_key: str) -> IngestionProgress | None:
        now = time.time()
        ttl = _ingestion_ttl_seconds()
        expired = [key for key, (_, created_at) in self._cache.items() if now - created_at > ttl]
        for key in expired:
            self._cache.pop(key, None)

        cached = self._cache.get(conversation_key)
        if not cached:
            return None
        return cached[0]

    def set(self, conversation_key: str, progress: IngestionProgress, ttl_seconds: int) -> None:
        self._cache[conversation_key] = (progress, time.time())

    def delete(self, conversation_key: str) -> None:
        self._cache.pop(conversation_key, None)

    def clear(self) -> None:
        self._cache.clear()


class RedisIngestionStore:
    def __init__(self, redis_url: str, key_prefix: str) -> None:
        if Redis is None:
            raise RedisError("redis package is not installed")
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix

    def _key(self, conversation_key: str) -> str:
        return f"{self._key_prefix}{conversation_key}"

    def get(self, conversation_key: str) -> IngestionProgress | None:
        value = self._client.get(self._key(conversation_key))
        if not value:
            return None
        try:
            payload = json.loads(value)
            return IngestionProgress(**payload)
        except (json.JSONDecodeError, TypeError):
            return None

    def set(self, conversation_key: str, progress: IngestionProgress, ttl_seconds: int) -> None:
        self._client.set(self._key(conversation_key), json.dumps(asdict(progress)), ex=ttl_seconds)

    def delete(self, conversation_key: str) -> None:
        try:
            self._client.delete(self._key(conversation_key))
        except RedisError:
            pass

    def clear(self) -> None:
        try:
            keys = self._client.keys(f"{self._key_prefix}*")
            if keys:
                self._client.delete(*keys)
        except RedisError:
            pass


_BACKEND: IngestionStoreBackend | None = None


def _ingestion_ttl_seconds() -> int:
    return max(60, int(os.getenv("TUBS_STAGED_INGEST_TTL_SECONDS", os.getenv("TUBS_THREAD_CACHE_TTL_SECONDS", "21600"))))


def _redis_url() -> str | None:
    value = os.getenv("REDIS_URL", "").strip()
    return value or None


def _backend_name() -> str:
    return os.getenv("TUBS_STAGED_INGEST_BACKEND", os.getenv("TUBS_THREAD_CACHE_BACKEND", "redis")).strip().lower()


def _key_prefix() -> str:
    return os.getenv("TUBS_STAGED_INGEST_PREFIX", "tubs:ingest:").strip() or "tubs:ingest:"


def _build_backend() -> IngestionStoreBackend:
    backend_name = _backend_name()
    if backend_name == "memory":
        return InMemoryIngestionStore()

    redis_url = _redis_url()
    if backend_name == "redis" and redis_url and Redis is not None:
        try:
            backend = RedisIngestionStore(redis_url, _key_prefix())
            backend._client.ping()
            return backend
        except RedisError:
            pass

    return InMemoryIngestionStore()


def _backend() -> IngestionStoreBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _build_backend()
    return _BACKEND


def get_ingestion_progress(conversation_key: str) -> IngestionProgress | None:
    try:
        return _backend().get(conversation_key)
    except RedisError:
        return None


def remember_ingestion_progress(conversation_key: str, progress: IngestionProgress) -> None:
    try:
        _backend().set(conversation_key, progress, _ingestion_ttl_seconds())
    except RedisError:
        pass


def forget_ingestion_progress(conversation_key: str) -> None:
    try:
        _backend().delete(conversation_key)
    except RedisError:
        pass


def reset_ingestion_progress() -> None:
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.clear()
    _BACKEND = None
