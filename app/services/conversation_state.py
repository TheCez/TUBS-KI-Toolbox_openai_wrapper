from __future__ import annotations

import hashlib
import math
import os
import re
import time
from typing import Any, Callable, Protocol, Sequence

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - exercised implicitly by environments without redis installed
    Redis = None

    class RedisError(Exception):
        pass

from app.services.translation import (
    extract_text_from_content,
    extract_tool_calls_from_content,
    extract_tool_results_from_content,
)


class ThreadCacheBackend(Protocol):
    def get(self, conversation_key: str) -> str | None: ...

    def set(self, conversation_key: str, thread_id: str, ttl_seconds: int) -> None: ...

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

    def clear(self) -> None:
        try:
            keys = self._client.keys(f"{self._key_prefix}*")
            if keys:
                self._client.delete(*keys)
        except RedisError:
            pass


_BACKEND: ThreadCacheBackend | None = None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate(text: str, max_chars: int) -> str:
    normalized = _normalize_whitespace(text)
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


def _tool_call_lines(content: Any) -> list[str]:
    calls = extract_tool_calls_from_content(content)
    return [
        f"assistant used `{tool_call['name']}` with {_truncate(tool_call['arguments'], 180)}"
        for tool_call in calls
        if tool_call.get("name")
    ]


def _tool_result_lines(content: Any) -> list[str]:
    results = extract_tool_results_from_content(content)
    lines = []
    for result in results:
        label = "tool error" if result["is_error"] else "tool result"
        if result["text"]:
            lines.append(f"{label}: {_truncate(result['text'], 220)}")
    return lines


def _summarize_message(message: Any) -> list[str]:
    role = _message_role(message)
    content = _message_content(message)

    lines: list[str] = []
    lines.extend(_tool_call_lines(content))
    lines.extend(_tool_result_lines(content))

    text = extract_text_from_content(content)
    if isinstance(content, str) and not text:
        text = content

    if text:
        label = role or "message"
        lines.append(f"{label}: {_truncate(text, 260)}")

    return lines


def _keep_last_turns() -> int:
    return max(1, int(os.getenv("TUBS_KEEP_LAST_TURNS", "8")))


def _summary_budget() -> int:
    return max(400, int(os.getenv("TUBS_COMPACT_SUMMARY_CHARS", "4000")))


def _thread_summary_budget() -> int:
    return max(200, int(os.getenv("TUBS_THREAD_SUMMARY_CHARS", "1200")))


def _max_prompt_tokens() -> int:
    return max(100, int(os.getenv("TUBS_MAX_PROMPT_TOKENS", "9000")))


def _thread_prompt_tokens() -> int:
    default_tokens = min(_max_prompt_tokens(), 3500)
    return max(50, int(os.getenv("TUBS_THREAD_PROMPT_TOKENS", str(default_tokens))))


def _chars_per_token() -> int:
    return max(2, int(os.getenv("TUBS_APPROX_CHARS_PER_TOKEN", "4")))


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


def _prompt_char_budget(thread_id: str | None) -> int:
    token_budget = _thread_prompt_tokens() if thread_id else _max_prompt_tokens()
    return token_budget * _chars_per_token()


def estimate_token_count(text: str) -> int:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / _chars_per_token()))


def _summary_header(thread_id: str | None) -> str:
    if thread_id:
        return "Thread context bridge:"
    return "Earlier conversation summary:"


def _build_summary(
    messages: Sequence[Any],
    *,
    summary_budget: int,
    thread_id: str | None,
) -> str | None:
    summary_lines: list[str] = []
    consumed = 0
    for message in messages:
        for line in _summarize_message(message):
            entry = f"- {line}"
            entry_len = len(entry) + 1
            if consumed + entry_len > summary_budget:
                summary_lines.append("- earlier context compacted to stay within the prompt budget")
                consumed = summary_budget
                break
            summary_lines.append(entry)
            consumed += entry_len
        if consumed >= summary_budget:
            break

    if not summary_lines:
        return None

    return _summary_header(thread_id) + "\n" + "\n".join(summary_lines)


def _keep_recent_indexes(messages: Sequence[Any], dialogue_indexes: Sequence[int], keep_last_turns: int) -> set[int]:
    if not dialogue_indexes:
        return set()

    keep_indexes = set(dialogue_indexes[-keep_last_turns:])
    last_user_index = None
    for idx in reversed(dialogue_indexes):
        if _message_role(messages[idx]) == "user":
            last_user_index = idx
            break

    if last_user_index is not None:
        keep_indexes.update(idx for idx in dialogue_indexes if idx >= last_user_index)

    return keep_indexes


def _recent_and_older_messages(
    messages: Sequence[Any],
    *,
    keep_last_turns: int,
) -> tuple[list[Any], list[Any]]:
    dialogue_indexes = [
        idx for idx, message in enumerate(messages) if _message_role(message) not in {"system", "developer"}
    ]
    keep_indexes = _keep_recent_indexes(messages, dialogue_indexes, keep_last_turns)
    recent_messages = [
        message
        for idx, message in enumerate(messages)
        if _message_role(message) in {"system", "developer"} or idx in keep_indexes
    ]
    older_messages = [
        message
        for idx, message in enumerate(messages)
        if _message_role(message) not in {"system", "developer"} and idx not in keep_indexes
    ]
    return recent_messages, older_messages


def _shrink_summary_to_fit(summary: str, max_chars: int, thread_id: str | None) -> str | None:
    if max_chars <= 0:
        return None
    if len(summary) <= max_chars:
        return summary

    body_budget = max_chars - len(_summary_header(thread_id)) - 1
    if body_budget <= 20:
        return None

    lines = summary.splitlines()[1:]
    kept_lines: list[str] = []
    consumed = 0
    for line in lines:
        entry_len = len(line) + 1
        if consumed + entry_len > body_budget:
            break
        kept_lines.append(line)
        consumed += entry_len

    if not kept_lines:
        return None
    if kept_lines[-1] != "- earlier context compacted to stay within the prompt budget":
        kept_lines.append("- earlier context compacted further for this request")
    return _summary_header(thread_id) + "\n" + "\n".join(kept_lines)


def _messages_as_compact_prompt(messages: Sequence[Any], max_chars: int) -> str:
    lines: list[str] = []
    consumed = 0
    for message in messages:
        for line in _summarize_message(message):
            entry = f"- {line}"
            entry_len = len(entry) + 1
            if consumed + entry_len > max_chars:
                if lines:
                    lines.append("- current request compacted to stay within the prompt budget")
                return "Compact current context:\n" + "\n".join(lines)
            lines.append(entry)
            consumed += entry_len
    return "Compact current context:\n" + "\n".join(lines)


def build_prompt_with_compaction(
    messages: Sequence[Any],
    *,
    compile_prompt: Callable[[Sequence[Any]], str],
    thread_id: str | None = None,
) -> str:
    keep_last_turns = _keep_last_turns()
    summary_budget = _thread_summary_budget() if thread_id else _summary_budget()
    prompt_budget = _prompt_char_budget(thread_id)

    dialogue_indexes = [
        idx for idx, message in enumerate(messages) if _message_role(message) not in {"system", "developer"}
    ]
    if not dialogue_indexes:
        return ""

    current_keep = min(len(dialogue_indexes), keep_last_turns)
    best_prompt = ""

    while current_keep >= 1:
        recent_messages, older_messages = _recent_and_older_messages(messages, keep_last_turns=current_keep)
        recent_prompt = compile_prompt(recent_messages).strip()
        summary = _build_summary(older_messages, summary_budget=summary_budget, thread_id=thread_id)
        candidate = prepend_summary_to_prompt(recent_prompt, summary).strip()

        if len(candidate) <= prompt_budget:
            return candidate

        available_for_summary = max(0, prompt_budget - len(recent_prompt) - 2)
        shrunken_summary = _shrink_summary_to_fit(summary or "", available_for_summary, thread_id)
        candidate = prepend_summary_to_prompt(recent_prompt, shrunken_summary).strip()
        if len(candidate) <= prompt_budget:
            return candidate

        if len(candidate) > len(best_prompt):
            best_prompt = candidate

        current_keep -= 1

    recent_messages, older_messages = _recent_and_older_messages(messages, keep_last_turns=1)
    fallback_summary = _build_summary(older_messages, summary_budget=min(summary_budget, 600), thread_id=thread_id)
    compact_recent = _messages_as_compact_prompt(recent_messages, max(120, prompt_budget // 2))
    fallback_prompt = prepend_summary_to_prompt(compact_recent, fallback_summary).strip()
    if len(fallback_prompt) <= prompt_budget:
        return fallback_prompt

    if len(compact_recent) <= prompt_budget:
        return compact_recent

    compact_text = _messages_as_compact_prompt(recent_messages, max(120, prompt_budget - 40))
    return compact_text[:prompt_budget].rstrip()


def compact_messages(messages: Sequence[Any]) -> tuple[list[Any], str | None]:
    keep_last_turns = _keep_last_turns()
    dialogue_indexes = [
        idx for idx, message in enumerate(messages) if _message_role(message) not in {"system", "developer"}
    ]
    if len(dialogue_indexes) <= keep_last_turns:
        return list(messages), None

    recent_messages, older_messages = _recent_and_older_messages(messages, keep_last_turns=keep_last_turns)
    return recent_messages, _build_summary(
        older_messages,
        summary_budget=_summary_budget(),
        thread_id=None,
    )


def prepend_summary_to_prompt(prompt: str, summary: str | None) -> str:
    if not summary:
        return prompt
    if not prompt:
        return summary
    return f"{summary}\n\n{prompt}"


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


def reset_thread_cache() -> None:
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.clear()
    _BACKEND = None
