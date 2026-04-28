from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Sequence

from app.models.context import CompactionArtifact, HotContextSnapshot
from app.services.context_chunking import summarize_text_chunks
from app.services.context_store import context_store
from app.services.debug_trace import record_debug_event
from app.services.tool_error_guidance import guidance_for_tool_errors
from app.services.translation import extract_text_from_content, extract_tool_results_from_content


_PATH_RE = re.compile(r"[A-Za-z]:\\[^\r\n]+|/[^\r\n]+")
_SYMBOL_RE = re.compile(r"\b(?:const|function|class|def|async def)\s+([A-Za-z_][A-Za-z0-9_]*)|<([A-Z][A-Za-z0-9_]*)\b")
_LABEL_RE = re.compile(r"^\s*[•\-\*]?\s*(name|my name|creature|vibe|emoji)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_EXACT_REPLY_RE = re.compile(r"reply(?: with exactly)?\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _text(message: Any) -> str:
    content = _message_content(message)
    text = extract_text_from_content(content)
    if isinstance(content, str) and not text:
        text = content
    return (text or "").strip()


def _extract_paths(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0).rstrip(") ") for match in _PATH_RE.finditer(text)))


def _extract_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for match in _SYMBOL_RE.finditer(text):
        if match.group(1):
            symbols.append(match.group(1))
        if match.group(2):
            symbols.append(match.group(2))
    return list(dict.fromkeys(symbols))


def _is_constraint_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["must ", "should ", "prefer ", "do not", "don't ", "cannot "])


def _is_goal_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["i want", "we want", "trying to", "need to", "goal", "objective"])


def _digest(thread_id: str, kind: str, summary: str) -> str:
    return hashlib.sha256(f"{thread_id}|{kind}|{summary}".encode("utf-8")).hexdigest()


def _identity_updates(text: str) -> dict[str, str]:
    updates: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LABEL_RE.match(line)
        if not match:
            continue
        field = match.group(1).strip().lower()
        value = match.group(2).strip()
        if field == "my name":
            updates["user_name"] = value
        elif field == "name":
            updates["assistant_name"] = value
        elif field == "creature":
            updates["assistant_creature"] = value
        elif field == "vibe":
            updates["assistant_vibe"] = value
        elif field == "emoji":
            updates["assistant_emoji"] = value
    return updates


def _bootstrap_prompt_detected(text: str) -> bool:
    lowered = text.lower()
    return (
        "pick anything you like for these four" in lowered
        or "mid-bootstrap" in lowered
        or "bootstrap-pending" in lowered
        or ("my name" in lowered and "creature" in lowered and "vibe" in lowered and "emoji" in lowered and "reply" in lowered)
    )


def _expected_reply(text: str) -> str | None:
    match = _EXACT_REPLY_RE.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if "\n" in value:
        value = value.splitlines()[0].strip()
    return value or None


def _build_hidden_bridge(latest_user_text: str | None, response_text: str | None, recent_messages: list[str]) -> str | None:
    bridge_parts: list[str] = []
    if latest_user_text:
        bridge_parts.append(f"Latest user intent: {latest_user_text[:220]}")
    if response_text:
        bridge_parts.append(f"Latest assistant direction: {response_text[:220]}")
    if recent_messages:
        bridge_parts.append("Recent turns: " + " | ".join(recent_messages[-3:]))
    if not bridge_parts:
        return None
    return "\n".join(bridge_parts)[:900]


class ContextIngestService:
    def __init__(self) -> None:
        self._store = context_store()

    def ingest_turn(
        self,
        thread_id: str,
        messages: Sequence[Any],
        response_text: str | None = None,
    ) -> None:
        candidate_records = []
        latest_user_text = None
        recent_messages: list[str] = []
        tool_failures: list[str] = []

        for index, message in enumerate(messages, start=1):
            role = _message_role(message)
            text = _text(message)
            if text:
                recent_messages.append(f"{role}: {text[:240]}")

            if role == "user" and text:
                latest_user_text = text
                candidate_records.append(
                    self._store.new_memory(
                        thread_id=thread_id,
                        kind="user_request",
                        title="User request",
                        content=text,
                        summary=text[:240],
                        source_turn_range=str(index),
                        importance=0.7,
                        recency_score=0.9,
                        file_paths=_extract_paths(text),
                        symbol_names=_extract_symbols(text),
                    )
                )
                if _is_goal_text(text):
                    candidate_records.append(
                        self._store.new_memory(
                            thread_id=thread_id,
                            kind="goal",
                            title="Current goal",
                            content=text,
                            summary=text[:200],
                            source_turn_range=str(index),
                            importance=0.9,
                            recency_score=0.9,
                            file_paths=_extract_paths(text),
                            symbol_names=_extract_symbols(text),
                        )
                    )
                if _is_constraint_text(text):
                    candidate_records.append(
                        self._store.new_memory(
                            thread_id=thread_id,
                            kind="constraint",
                            title="Constraint or preference",
                            content=text,
                            summary=text[:200],
                            source_turn_range=str(index),
                            importance=0.8,
                            recency_score=0.8,
                            file_paths=_extract_paths(text),
                            symbol_names=_extract_symbols(text),
                        )
                    )

            tool_results = extract_tool_results_from_content(_message_content(message))
            if role == "tool" and text:
                tool_results = [{"id": "", "text": text, "is_error": False, "type": "tool"}]
            for result in tool_results:
                if not result.get("text"):
                    continue
                if result.get("is_error"):
                    tool_failures.append(result["text"])
                    guidance = guidance_for_tool_errors([result])
                    combined = result["text"]
                    if guidance:
                        combined = f"{combined}\n" + "\n".join(guidance)
                    candidate_records.append(
                        self._store.new_memory(
                            thread_id=thread_id,
                            kind="tool_failure",
                            title="Tool failure",
                            content=combined,
                            summary=result["text"][:220],
                            source_turn_range=str(index),
                            source_tool=result.get("id") or result.get("type"),
                            importance=0.9,
                            recency_score=0.9,
                            file_paths=_extract_paths(result["text"]),
                            symbol_names=_extract_symbols(result["text"]),
                        )
                    )

        if response_text:
            candidate_records.append(
                self._store.new_memory(
                    thread_id=thread_id,
                    kind="assistant_response",
                    title="Assistant response",
                    content=response_text,
                    summary=response_text[:220],
                    source_turn_range="assistant",
                    importance=0.4,
                    recency_score=0.7,
                    file_paths=_extract_paths(response_text),
                    symbol_names=_extract_symbols(response_text),
                )
            )

        unique_records = self._dedupe(thread_id, candidate_records)
        if unique_records:
            self._store.upsert_memories(unique_records)

        self._update_hot_snapshot(
            thread_id=thread_id,
            latest_user_text=latest_user_text,
            tool_failures=tool_failures,
            recent_messages=recent_messages,
            response_text=response_text,
        )

    def _dedupe(self, thread_id: str, records: list) -> list:
        recent = self._store.recent(thread_id, 100)
        seen = {_digest(thread_id, record.kind, record.summary) for record in recent}
        unique = []
        for record in records:
            signature = _digest(thread_id, record.kind, record.summary)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(record)
        return unique

    def _update_hot_snapshot(
        self,
        *,
        thread_id: str,
        latest_user_text: str | None,
        tool_failures: list[str],
        recent_messages: list[str],
        response_text: str | None,
    ) -> None:
        snapshot = self._store.get_hot_snapshot(thread_id)
        now = datetime.now(UTC)
        if snapshot is None:
            snapshot = HotContextSnapshot(thread_id=thread_id, updated_at=now)

        if latest_user_text:
            snapshot.current_objective = latest_user_text[:300]
            if latest_user_text.lower().startswith(("1.", "- ", "* ")):
                snapshot.current_plan = [part for part in summarize_text_chunks(latest_user_text, per_chunk_chars=120, max_chunks=4)]
            snapshot.active_workflow.current_goal = latest_user_text[:300]
            if snapshot.active_workflow.kind is None:
                snapshot.active_workflow.kind = "general_chat"
            if snapshot.active_workflow.status is None:
                snapshot.active_workflow.status = "active"
            snapshot.task_state.in_progress_tasks = list(
                dict.fromkeys(snapshot.task_state.in_progress_tasks + [snapshot.active_workflow.kind or "general_chat"])
            )[-6:]

        if tool_failures:
            merged_failures = snapshot.latest_tool_failures + [item[:240] for item in tool_failures]
            snapshot.latest_tool_failures = list(dict.fromkeys(merged_failures))[-4:]
            snapshot.unresolved_blockers = snapshot.latest_tool_failures[:]
            snapshot.active_workflow.blocked_on_user = False
            snapshot.active_workflow.status = "blocked_on_error"

        if response_text and any(token in response_text.lower() for token in ["decide", "recommend", "should", "will"]):
            snapshot.recent_decisions = list(dict.fromkeys(snapshot.recent_decisions + [response_text[:220]]))[-4:]
        if response_text and any(token in response_text.lower() for token in ["done", "completed", "working now", "succeeded"]):
            active_kind = snapshot.active_workflow.kind or "general_chat"
            snapshot.task_state.completed_tasks = list(dict.fromkeys(snapshot.task_state.completed_tasks + [active_kind]))[-8:]
            snapshot.task_state.in_progress_tasks = [item for item in snapshot.task_state.in_progress_tasks if item != active_kind]
            snapshot.active_workflow.status = "completed"

        combined_texts = [text for text in [latest_user_text, response_text] if text]
        for text in combined_texts:
            updates = _identity_updates(text)
            if updates.get("user_name"):
                snapshot.user_identity.name = updates["user_name"]
            if updates.get("assistant_name"):
                snapshot.assistant_identity.name = updates["assistant_name"]
            if updates.get("assistant_creature"):
                snapshot.assistant_identity.creature = updates["assistant_creature"]
            if updates.get("assistant_vibe"):
                snapshot.assistant_identity.vibe = updates["assistant_vibe"]
            if updates.get("assistant_emoji"):
                snapshot.assistant_identity.emoji = updates["assistant_emoji"]
            if _bootstrap_prompt_detected(text):
                snapshot.bootstrap_state.status = "pending"
                snapshot.bootstrap_state.pending_step = "identity_confirmation"
                snapshot.bootstrap_state.required_fields = ["name", "creature", "vibe", "emoji"]
                expected = _expected_reply(text)
                if expected:
                    snapshot.bootstrap_state.last_exact_expected_reply = expected
                snapshot.active_workflow.kind = "bootstrap"
                snapshot.active_workflow.status = "awaiting_identity"
                snapshot.active_workflow.blocked_on_user = True

        answered_fields: list[str] = []
        if snapshot.assistant_identity.name:
            answered_fields.append("name")
        if snapshot.assistant_identity.creature:
            answered_fields.append("creature")
        if snapshot.assistant_identity.vibe:
            answered_fields.append("vibe")
        if snapshot.assistant_identity.emoji:
            answered_fields.append("emoji")
        snapshot.bootstrap_state.answered_fields = answered_fields
        if answered_fields and snapshot.bootstrap_state.status in {"pending", "unknown"}:
            snapshot.bootstrap_state.status = "answered"
        if len(answered_fields) == 4:
            snapshot.bootstrap_state.status = "completed"
            snapshot.bootstrap_state.pending_step = None
            snapshot.active_workflow.kind = "general_chat"
            snapshot.active_workflow.status = "active"
            snapshot.active_workflow.blocked_on_user = False

        snapshot.recent_messages = recent_messages[-6:]
        snapshot.hidden_bridge_summary = _build_hidden_bridge(latest_user_text, response_text, snapshot.recent_messages)
        if snapshot.hidden_bridge_summary:
            snapshot.compaction_artifacts.append(
                CompactionArtifact(
                    artifact_id=f"bridge_{uuid.uuid4().hex}",
                    source_turn_range="latest",
                    kind="bridge",
                    summary=snapshot.hidden_bridge_summary,
                    workflow_kind=snapshot.active_workflow.kind,
                    workflow_status=snapshot.active_workflow.status,
                    created_at=now,
                )
            )
            snapshot.compaction_artifacts = snapshot.compaction_artifacts[-5:]
        snapshot.updated_at = now
        self._store.set_hot_snapshot(snapshot)
        record_debug_event(
            thread_id,
            "context_ingested",
            {
                "objective": snapshot.current_objective,
                "workflow_kind": snapshot.active_workflow.kind,
                "workflow_status": snapshot.active_workflow.status,
                "bootstrap_status": snapshot.bootstrap_state.status,
                "user_name": snapshot.user_identity.name,
                "assistant_name": snapshot.assistant_identity.name,
            },
        )


def context_ingest_service() -> ContextIngestService:
    return ContextIngestService()
