from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from fastapi import HTTPException

from app.services.conversation_state import forget_thread_id
from app.services.context_store import context_store
from app.services.staged_ingestion_store import forget_ingestion_progress


T = TypeVar("T")


_CONVERSATION_LIMIT_MARKERS = (
    "token limit für dieses gespräch überschritten",
    "token limit for this conversation",
    "conversation token limit",
    "context length exceeded for this conversation",
)


def _detail_text(detail: object) -> str:
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    return str(detail)


def is_thread_exhaustion_error(exc: HTTPException) -> bool:
    detail = _detail_text(exc.detail).lower()
    return any(marker in detail for marker in _CONVERSATION_LIMIT_MARKERS)


def reset_upstream_thread_state(conversation_key: str) -> None:
    forget_thread_id(conversation_key)
    forget_ingestion_progress(conversation_key)
    snapshot = context_store().get_hot_snapshot(conversation_key)
    if snapshot is not None:
        snapshot.thread_control.rotation_count += 1
        snapshot.thread_control.upstream_thread_id = None
        context_store().set_hot_snapshot(snapshot)


async def retry_with_fresh_thread_on_limit(
    conversation_key: str,
    action: Callable[[bool], Awaitable[T]],
) -> T:
    try:
        return await action(False)
    except HTTPException as exc:
        if not is_thread_exhaustion_error(exc):
            raise
        reset_upstream_thread_state(conversation_key)
        return await action(True)
