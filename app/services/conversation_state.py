from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any, Iterable, Sequence

from app.services.translation import (
    extract_text_from_content,
    extract_tool_calls_from_content,
    extract_tool_results_from_content,
)

_THREAD_CACHE: dict[str, tuple[str, float]] = {}


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
        f"assistant used `{tool_call['name']}` with { _truncate(tool_call['arguments'], 180) }"
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


def compact_messages(messages: Sequence[Any]) -> tuple[list[Any], str | None]:
    keep_last_turns = _keep_last_turns()
    summary_budget = _summary_budget()

    dialogue_indexes = [
        idx for idx, message in enumerate(messages) if _message_role(message) not in {"system", "developer"}
    ]
    if len(dialogue_indexes) <= keep_last_turns:
        return list(messages), None

    keep_indexes = set(dialogue_indexes[-keep_last_turns:])
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

    summary_lines: list[str] = []
    consumed = 0
    for message in older_messages:
        for line in _summarize_message(message):
            entry = f"- {line}"
            entry_len = len(entry) + 1
            if consumed + entry_len > summary_budget:
                summary_lines.append("- earlier context truncated to stay within the token budget")
                consumed = summary_budget
                break
            summary_lines.append(entry)
            consumed += entry_len
        if consumed >= summary_budget:
            break

    if not summary_lines:
        return recent_messages, None

    summary = "Earlier conversation summary:\n" + "\n".join(summary_lines)
    return recent_messages, summary


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


def _thread_cache_ttl_seconds() -> int:
    return max(60, int(os.getenv("TUBS_THREAD_CACHE_TTL_SECONDS", "21600")))


def _prune_thread_cache() -> None:
    now = time.time()
    ttl = _thread_cache_ttl_seconds()
    expired = [key for key, (_, created_at) in _THREAD_CACHE.items() if now - created_at > ttl]
    for key in expired:
        _THREAD_CACHE.pop(key, None)


def get_cached_thread_id(conversation_key: str) -> str | None:
    _prune_thread_cache()
    cached = _THREAD_CACHE.get(conversation_key)
    if not cached:
        return None
    return cached[0]


def remember_thread_id(conversation_key: str, response_payload: dict[str, Any] | None) -> None:
    if not response_payload:
        return
    thread = response_payload.get("thread")
    if isinstance(thread, dict):
        thread_id = thread.get("id")
    else:
        thread_id = thread
    if isinstance(thread_id, str) and thread_id.strip():
        _THREAD_CACHE[conversation_key] = (thread_id.strip(), time.time())


def reset_thread_cache() -> None:
    _THREAD_CACHE.clear()
