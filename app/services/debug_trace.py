from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover
    Redis = None

    class RedisError(Exception):
        pass


_MEMORY: dict[str, list[dict[str, Any]]] = {}


def _enabled() -> bool:
    return os.getenv("TUBS_DEBUG_TRACE_ENABLED", "true").strip().lower() == "true"


def _max_events() -> int:
    return max(10, int(os.getenv("TUBS_DEBUG_TRACE_MAX_EVENTS", "60")))


def _redis_client():
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url or Redis is None:
        return None
    try:
        client = Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except RedisError:
        return None


def _key(thread_id: str) -> str:
    prefix = os.getenv("TUBS_DEBUG_TRACE_PREFIX", "tubs:trace:").strip() or "tubs:trace:"
    return f"{prefix}{thread_id}"


def record_debug_event(thread_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    if not _enabled():
        return
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "payload": payload or {},
    }
    client = _redis_client()
    if client is not None:
        try:
            client.rpush(_key(thread_id), json.dumps(entry, ensure_ascii=False))
            client.ltrim(_key(thread_id), -_max_events(), -1)
            return
        except RedisError:
            pass
    bucket = _MEMORY.setdefault(thread_id, [])
    bucket.append(entry)
    if len(bucket) > _max_events():
        del bucket[:-_max_events()]


def get_debug_trace(thread_id: str) -> list[dict[str, Any]]:
    client = _redis_client()
    if client is not None:
        try:
            values = client.lrange(_key(thread_id), 0, -1)
            return [json.loads(item) for item in values]
        except (RedisError, json.JSONDecodeError):
            return []
    return list(_MEMORY.get(thread_id, []))
