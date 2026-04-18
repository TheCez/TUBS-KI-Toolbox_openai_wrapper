from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Sequence

from app.models.tubs import TubsChatRequest
from app.services.context_chunking import split_text_semantically, summarize_text_chunks
from app.services.context_compaction import estimate_token_count
from app.services.model_map import resolve_model
from app.services.staged_ingestion_store import (
    IngestionProgress,
    get_ingestion_progress,
    remember_ingestion_progress,
)
from app.services.translation import extract_text_from_content
from app.services.tubs_client import async_send_tubs_request


@dataclass
class StagedIngestionResult:
    thread_id: str | None
    messages: list[Any]
    applied: bool
    total_blocks: int = 0


def _enabled() -> bool:
    return os.getenv("TUBS_ENABLE_STAGED_INGESTION", "false").strip().lower() == "true"


def _threshold_tokens() -> int:
    return max(20, int(os.getenv("TUBS_STAGED_INGEST_THRESHOLD_TOKENS", "7000")))


def _block_tokens() -> int:
    return max(10, int(os.getenv("TUBS_STAGED_INGEST_BLOCK_TOKENS", "2200")))


def _chars_per_token() -> int:
    return max(2, int(os.getenv("TUBS_APPROX_CHARS_PER_TOKEN", "4")))


def _max_blocks() -> int:
    return max(2, int(os.getenv("TUBS_STAGED_INGEST_MAX_BLOCKS", "12")))


def _ledger_chars() -> int:
    return max(200, int(os.getenv("TUBS_STAGED_INGEST_LEDGER_CHARS", "1200")))


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _has_non_text_content(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
        else:
            item_type = getattr(item, "type", None)
        if item_type not in {None, "text", "input_text", "output_text"}:
            return True
    return False


def _copy_with_new_content(message: Any, content: str) -> Any:
    if isinstance(message, dict):
        updated = dict(message)
        updated["content"] = content
        return updated
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    setattr(message, "content", content)
    return message


def _latest_user_text(messages: Sequence[Any]) -> tuple[int | None, str | None]:
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if _message_role(message) != "user":
            continue
        content = _message_content(message)
        if _has_non_text_content(content):
            return None, None
        text = extract_text_from_content(content)
        if isinstance(content, str) and not text:
            text = content
        text = (text or "").strip()
        if text:
            return idx, text
    return None, None


def _split_blocks(text: str) -> list[str]:
    max_chars = _block_tokens() * _chars_per_token()
    blocks = split_text_semantically(text, max_chars)
    if len(blocks) <= _max_blocks():
        return blocks

    # Increase block size rather than dropping content if the initial split is too fine.
    scale = max(1, len(blocks) // _max_blocks())
    larger_chars = max_chars * (scale + 1)
    blocks = split_text_semantically(text, larger_chars)
    if len(blocks) <= _max_blocks():
        return blocks

    return blocks[: _max_blocks() - 1] + ["\n\n".join(blocks[_max_blocks() - 1 :])]


def _block_summaries(blocks: Sequence[str]) -> list[str]:
    summaries: list[str] = []
    for idx, block in enumerate(blocks, start=1):
        summary = summarize_text_chunks(block, per_chunk_chars=220, max_chunks=1)
        label = summary[0] if summary else f"context block {idx}"
        summaries.append(f"{idx}/{len(blocks)}: {label}")
    return summaries


def _ingestion_signature(model: Any, text: str) -> str:
    raw_model = str(model.value) if hasattr(model, "value") else str(model)
    return hashlib.sha256(f"{raw_model}||{text}".encode("utf-8")).hexdigest()


def _ingestion_custom_instructions() -> str:
    return (
        "Context ingestion mode. Retain the provided block for later reasoning. "
        "Do not solve the task yet. Do not call tools. Reply only with ACK."
    )


def _ledger_text(summaries: Sequence[str]) -> str:
    joined = "\n".join(f"- {summary}" for summary in summaries)
    if len(joined) <= _ledger_chars():
        return joined
    return joined[: _ledger_chars() - 3].rstrip() + "..."


def _ingestion_prompt(block_number: int, total_blocks: int, block_summary: str, block_text: str) -> str:
    return (
        f"[User]: Context block {block_number}/{total_blocks} for later reasoning.\n"
        f"[User]: Block summary: {block_summary}\n"
        "[User]: Retain this information exactly for the final task. Do not solve it yet.\n"
        f"[User]:\n{block_text}"
    )


def _final_reference_text(total_blocks: int, summaries: Sequence[str]) -> str:
    ledger = _ledger_text(summaries)
    latest_summary = summaries[-1] if summaries else "latest block summary unavailable"
    return (
        "The detailed latest request was pre-ingested into this thread in staged context blocks.\n"
        f"Available blocks: 1/{total_blocks} through {total_blocks}/{total_blocks}.\n"
        "Use those ingested blocks as the authoritative detailed context for this turn.\n"
        f"Block ledger:\n{ledger}\n"
        f"Most recent block summary: {latest_summary}\n"
        "Now answer or act on the user's request using the ingested blocks."
    )


async def prepare_staged_messages(
    *,
    model: Any,
    messages: Sequence[Any],
    thread_id: str | None,
    conversation_key: str,
    bearer_token: str,
) -> StagedIngestionResult:
    if not _enabled():
        return StagedIngestionResult(thread_id=thread_id, messages=list(messages), applied=False)

    latest_index, latest_text = _latest_user_text(messages)
    if latest_index is None or not latest_text:
        return StagedIngestionResult(thread_id=thread_id, messages=list(messages), applied=False)

    if estimate_token_count(latest_text) < _threshold_tokens():
        return StagedIngestionResult(thread_id=thread_id, messages=list(messages), applied=False)

    blocks = _split_blocks(latest_text)
    if len(blocks) <= 1:
        return StagedIngestionResult(thread_id=thread_id, messages=list(messages), applied=False)

    summaries = _block_summaries(blocks)
    signature = _ingestion_signature(model, latest_text)
    progress = get_ingestion_progress(conversation_key)
    current_thread_id = thread_id
    completed_blocks = 0
    if progress and progress.signature == signature and progress.total_blocks == len(blocks):
        completed_blocks = min(progress.completed_blocks, len(blocks))
        current_thread_id = progress.thread_id or current_thread_id

    model_name = resolve_model(model)
    for block_index in range(completed_blocks, len(blocks)):
        payload = TubsChatRequest(
            thread=current_thread_id,
            prompt=_ingestion_prompt(block_index + 1, len(blocks), summaries[block_index], blocks[block_index]),
            model=model_name,
            customInstructions=_ingestion_custom_instructions(),
        ).model_dump(exclude_none=True)
        response_payload = await async_send_tubs_request(
            payload=payload,
            images=[],
            bearer_token=bearer_token,
            stream=False,
        )
        thread = response_payload.get("thread")
        if isinstance(thread, dict):
            current_thread_id = thread.get("id", current_thread_id)
        elif isinstance(thread, str) and thread.strip():
            current_thread_id = thread.strip()

        remember_ingestion_progress(
            conversation_key,
            IngestionProgress(
                signature=signature,
                total_blocks=len(blocks),
                completed_blocks=block_index + 1,
                thread_id=current_thread_id,
            ),
        )

    updated_messages = list(messages)
    updated_messages[latest_index] = _copy_with_new_content(
        updated_messages[latest_index],
        _final_reference_text(len(blocks), summaries),
    )
    return StagedIngestionResult(
        thread_id=current_thread_id,
        messages=updated_messages,
        applied=True,
        total_blocks=len(blocks),
    )
