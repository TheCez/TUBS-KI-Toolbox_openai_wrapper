from __future__ import annotations

import math
import os
from typing import Any, Callable, Sequence

from app.services.context_chunking import normalize_whitespace, summarize_text_chunks
from app.services.translation import (
    extract_text_from_content,
    extract_tool_calls_from_content,
    extract_tool_results_from_content,
)


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


def _keep_last_turns() -> int:
    return max(1, int(os.getenv("TUBS_KEEP_LAST_TURNS", "8")))


def _summary_budget() -> int:
    return max(400, int(os.getenv("TUBS_COMPACT_SUMMARY_CHARS", "4000")))


def _thread_summary_budget() -> int:
    return max(200, int(os.getenv("TUBS_THREAD_SUMMARY_CHARS", "1200")))


def _max_prompt_tokens() -> int:
    return max(100, int(os.getenv("TUBS_MAX_PROMPT_TOKENS", "9000")))


def _thread_prompt_tokens() -> int:
    configured = os.getenv("TUBS_THREAD_PROMPT_TOKENS")
    if configured is None or not configured.strip():
        return _max_prompt_tokens()
    return max(50, int(configured))


def _chars_per_token() -> int:
    return max(2, int(os.getenv("TUBS_APPROX_CHARS_PER_TOKEN", "4")))


def _prompt_char_budget(thread_id: str | None) -> int:
    token_budget = _thread_prompt_tokens() if thread_id else _max_prompt_tokens()
    return token_budget * _chars_per_token()


def estimate_token_count(text: str) -> int:
    normalized = normalize_whitespace(text)
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / _chars_per_token()))


def _summary_header(thread_id: str | None) -> str:
    if thread_id:
        return "Thread context bridge:"
    return "Earlier conversation summary:"


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


def _summarize_message(
    message: Any,
    *,
    text_chunk_chars: int = 260,
    max_text_chunks: int = 3,
) -> list[str]:
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
        for chunk in summarize_text_chunks(text, per_chunk_chars=text_chunk_chars, max_chunks=max_text_chunks):
            lines.append(f"{label}: {chunk}")

    return lines


def _summary_lines_for_messages(
    messages: Sequence[Any],
    *,
    summary_budget: int,
    text_chunk_chars: int = 220,
    max_text_chunks: int = 2,
) -> list[str]:
    summary_lines: list[str] = []
    consumed = 0
    for message in messages:
        for line in _summarize_message(
            message,
            text_chunk_chars=text_chunk_chars,
            max_text_chunks=max_text_chunks,
        ):
            entry = f"- {line}"
            entry_len = len(entry) + 1
            if consumed + entry_len > summary_budget:
                summary_lines.append("- earlier context compacted to stay within the prompt budget")
                return summary_lines
            summary_lines.append(entry)
            consumed += entry_len
    return summary_lines


def _build_summary(
    messages: Sequence[Any],
    *,
    summary_budget: int,
    thread_id: str | None,
) -> str | None:
    summary_lines = _summary_lines_for_messages(messages, summary_budget=summary_budget)
    if not summary_lines:
        return None
    return _summary_header(thread_id) + "\n" + "\n".join(summary_lines)


def _append_summary_lines(
    existing_lines: list[str],
    additional_messages: Sequence[Any],
    *,
    summary_budget: int,
) -> list[str]:
    merged = list(existing_lines)
    consumed = sum(len(line) + 1 for line in merged)

    if merged and merged[-1] == "- earlier context compacted to stay within the prompt budget":
        merged.pop()
        consumed = sum(len(line) + 1 for line in merged)

    for message in additional_messages:
        for line in _summarize_message(message, text_chunk_chars=220, max_text_chunks=2):
            entry = f"- {line}"
            entry_len = len(entry) + 1
            if consumed + entry_len > summary_budget:
                merged.append("- earlier context compacted to stay within the prompt budget")
                return merged
            merged.append(entry)
            consumed += entry_len

    return merged


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
    text_chunk_chars = min(320, max(120, max_chars // 3))
    for message in messages:
        for line in _summarize_message(
            message,
            text_chunk_chars=text_chunk_chars,
            max_text_chunks=4,
        ):
            entry = f"- {line}"
            entry_len = len(entry) + 1
            if consumed + entry_len > max_chars:
                if lines:
                    lines.append("- current request compacted to stay within the prompt budget")
                return "Compact current context:\n" + "\n".join(lines)
            lines.append(entry)
            consumed += entry_len
    return "Compact current context:\n" + "\n".join(lines)


def _summary_from_lines(summary_lines: Sequence[str], *, thread_id: str | None) -> str | None:
    if not summary_lines:
        return None
    return _summary_header(thread_id) + "\n" + "\n".join(summary_lines)


def _foldable_dialogue_indexes(messages: Sequence[Any]) -> list[int]:
    dialogue_indexes = [
        idx for idx, message in enumerate(messages) if _message_role(message) not in {"system", "developer"}
    ]
    if len(dialogue_indexes) <= 1:
        return []

    protected_start = dialogue_indexes[-1]
    for idx in reversed(dialogue_indexes):
        if _message_role(messages[idx]) == "user":
            protected_start = idx
            break

    return [idx for idx in dialogue_indexes if idx < protected_start]


def _compact_oldest_block(
    messages: Sequence[Any],
    *,
    chunk_size: int,
) -> tuple[list[Any], list[Any]]:
    foldable_indexes = _foldable_dialogue_indexes(messages)
    if not foldable_indexes:
        return list(messages), []

    selected_indexes = set(foldable_indexes[:chunk_size])
    folded_messages = [message for idx, message in enumerate(messages) if idx in selected_indexes]
    remaining_messages = [message for idx, message in enumerate(messages) if idx not in selected_indexes]
    return remaining_messages, folded_messages


def prepend_summary_to_prompt(prompt: str, summary: str | None) -> str:
    if not summary:
        return prompt
    if not prompt:
        return summary
    return f"{summary}\n\n{prompt}"


def build_prompt_with_compaction(
    messages: Sequence[Any],
    *,
    compile_prompt: Callable[[Sequence[Any]], str],
    thread_id: str | None = None,
    prompt_token_budget: int | None = None,
) -> str:
    keep_last_turns = _keep_last_turns()
    summary_budget = _thread_summary_budget() if thread_id else _summary_budget()
    if prompt_token_budget is None:
        prompt_budget = _prompt_char_budget(thread_id)
    else:
        prompt_budget = max(80, prompt_token_budget) * _chars_per_token()
    step_chunk_size = max(2, keep_last_turns)

    full_prompt = compile_prompt(messages).strip()
    if len(full_prompt) <= prompt_budget:
        return full_prompt

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

    working_messages = list(messages)
    rolling_summary_lines: list[str] = []
    while True:
        prompt_text = compile_prompt(working_messages).strip()
        summary_text = _summary_from_lines(rolling_summary_lines, thread_id=thread_id)
        candidate = prepend_summary_to_prompt(prompt_text, summary_text).strip()
        if len(candidate) <= prompt_budget:
            return candidate

        next_messages, folded_messages = _compact_oldest_block(working_messages, chunk_size=step_chunk_size)
        if not folded_messages:
            break

        rolling_summary_lines = _append_summary_lines(
            rolling_summary_lines,
            folded_messages,
            summary_budget=summary_budget,
        )
        working_messages = next_messages
        if len(candidate) > len(best_prompt):
            best_prompt = candidate

    recent_messages, older_messages = _recent_and_older_messages(working_messages, keep_last_turns=1)
    fallback_summary_lines = _append_summary_lines(
        rolling_summary_lines,
        older_messages,
        summary_budget=min(summary_budget, 600),
    )
    fallback_summary = _summary_from_lines(fallback_summary_lines, thread_id=thread_id)
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
