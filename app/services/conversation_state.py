from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Protocol, Sequence

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - exercised implicitly by environments without redis installed
    Redis = None

    class RedisError(Exception):
        pass

from app.services.context_compaction import (
    build_prompt_with_compaction,
    compact_messages,
    estimate_token_count,
    prepend_summary_to_prompt,
)
from app.services.context_chunking import normalize_whitespace
from app.services.translation import extract_text_from_content


class ThreadCacheBackend(Protocol):
    def get(self, conversation_key: str) -> str | None: ...

    def set(self, conversation_key: str, thread_id: str, ttl_seconds: int) -> None: ...

    def delete(self, conversation_key: str) -> None: ...

    def clear(self) -> None: ...


class InMemoryThreadCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, float]] = {}

    def get(self, conversation_key: str) -> str | None:
        now = time.time()
        ttl = _thread_cache_ttl_seconds()
        expired = [key for key, (_, created_at) in self._cache.items() if now - created_at > ttl]
        for key in expired:
            self._cache.pop(key, None)

        cached = self._cache.get(conversation_key)
        if not cached:
            return None
        return cached[0]

    def set(self, conversation_key: str, thread_id: str, ttl_seconds: int) -> None:
        self._cache[conversation_key] = (thread_id, time.time())

    def delete(self, conversation_key: str) -> None:
        self._cache.pop(conversation_key, None)

    def clear(self) -> None:
        self._cache.clear()


class RedisThreadCache:
    def __init__(self, redis_url: str, key_prefix: str) -> None:
        if Redis is None:
            raise RedisError("redis package is not installed")
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix

    def _key(self, conversation_key: str) -> str:
        return f"{self._key_prefix}{conversation_key}"

    def get(self, conversation_key: str) -> str | None:
        value = self._client.get(self._key(conversation_key))
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def set(self, conversation_key: str, thread_id: str, ttl_seconds: int) -> None:
        self._client.set(self._key(conversation_key), thread_id, ex=ttl_seconds)

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


_BACKEND: ThreadCacheBackend | None = None


def _truncate(text: str, max_chars: int) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _thread_cache_ttl_seconds() -> int:
    return max(60, int(os.getenv("TUBS_THREAD_CACHE_TTL_SECONDS", "21600")))


def _thread_cache_backend_name() -> str:
    return os.getenv("TUBS_THREAD_CACHE_BACKEND", "redis").strip().lower()


def _thread_cache_prefix() -> str:
    return os.getenv("TUBS_THREAD_CACHE_PREFIX", "tubs:thread:").strip() or "tubs:thread:"


def _redis_url() -> str | None:
    value = os.getenv("REDIS_URL", "").strip()
    return value or None


def _build_backend() -> ThreadCacheBackend:
    backend_name = _thread_cache_backend_name()
    if backend_name == "memory":
        return InMemoryThreadCache()

    redis_url = _redis_url()
    if backend_name == "redis" and redis_url and Redis is not None:
        try:
            backend = RedisThreadCache(redis_url, _thread_cache_prefix())
            backend._client.ping()
            return backend
        except RedisError:
            pass

    return InMemoryThreadCache()


def _backend() -> ThreadCacheBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _build_backend()
    return _BACKEND


def build_conversation_key(
    *,
    bearer_token: str,
    model: str,
    messages: Sequence[Any],
    explicit_user: str | None = None,
) -> str:
    anchor_parts = [explicit_user or "", model]
    included_roles: set[str] = set()
    for message in messages:
        role = _message_role(message)
        if role not in {"system", "developer", "user"}:
            continue
        if role in included_roles:
            continue
        text = extract_text_from_content(_message_content(message))
        if isinstance(_message_content(message), str) and not text:
            text = _message_content(message)
        if text:
            anchor_parts.append(f"{role}:{_truncate(text, 160)}")
            included_roles.add(role)
        if len(anchor_parts) >= 4:
            break

    digest_source = f"{hashlib.sha256(bearer_token.encode('utf-8')).hexdigest()}||" + "||".join(anchor_parts)
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()


def get_cached_thread_id(conversation_key: str) -> str | None:
    try:
        return _backend().get(conversation_key)
    except RedisError:
        return None


def remember_thread_id(conversation_key: str, response_payload: dict[str, Any] | None) -> None:
    if not response_payload:
        return
    thread = response_payload.get("thread")
    if isinstance(thread, dict):
        thread_id = thread.get("id")
    else:
        thread_id = thread
    if isinstance(thread_id, str) and thread_id.strip():
        try:
            _backend().set(conversation_key, thread_id.strip(), _thread_cache_ttl_seconds())
        except RedisError:
            pass


def forget_thread_id(conversation_key: str) -> None:
    try:
        _backend().delete(conversation_key)
    except RedisError:
        pass


def reset_thread_cache() -> None:
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.clear()
    _BACKEND = None
