from __future__ import annotations

import json
import os
from typing import Any, Iterable

from app.models.context import (
    GetContextByIdsArgs,
    GetThreadStateArgs,
    SearchContextArgs,
    StoreContextNoteArgs,
    SummarizeContextWindowArgs,
)
from app.services.context_store import context_store


def _enabled() -> bool:
    return os.getenv("TUBS_CONTEXT_TOOLS_ENABLED", "true").strip().lower() == "true"


def _tool_names() -> set[str]:
    return {
        "search_context",
        "get_context_by_ids",
        "get_thread_state",
        "store_context_note",
        "summarize_context_window",
    }


def _search_top_k_max() -> int:
    return max(1, int(os.getenv("TUBS_CONTEXT_SEARCH_TOP_K_MAX", "4")))


def _summary_chars() -> int:
    return max(80, int(os.getenv("TUBS_CONTEXT_SUMMARY_CHARS", "280")))


def _record_content_chars() -> int:
    return max(120, int(os.getenv("TUBS_CONTEXT_RECORD_CONTENT_CHARS", "500")))


def _get_by_ids_max_records() -> int:
    return max(1, int(os.getenv("TUBS_CONTEXT_GET_BY_IDS_MAX_RECORDS", "3")))


def _thread_state_recent_messages_max() -> int:
    return max(0, int(os.getenv("TUBS_CONTEXT_THREAD_STATE_RECENT_MESSAGES_MAX", "2")))


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def is_context_tool(name: str) -> bool:
    return name in _tool_names()


def context_tools_for_openai() -> list[dict[str, Any]]:
    if not _enabled():
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "search_context",
                "description": "Optionally search durable thread context using semantic RAG-style retrieval plus exact filters when prior goals, file facts, failures, or decisions may matter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "kinds": {"type": "array", "items": {"type": "string"}},
                        "file_paths": {"type": "array", "items": {"type": "string"}},
                        "symbols": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_context_by_ids",
                "description": "Fetch exact durable context records by ID after a prior semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_thread_state",
                "description": "Get the current hot context snapshot for the active thread, including objective, plan, blockers, and recent decisions.",
                "parameters": {
                    "type": "object",
                    "properties": {"include_recent_messages": {"type": "boolean"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "store_context_note",
                "description": "Store an explicit durable context note for later retrieval when a result or decision should be remembered.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "kind": {"type": "string"},
                        "file_paths": {"type": "array", "items": {"type": "string"}},
                        "symbol_names": {"type": "array", "items": {"type": "string"}},
                        "importance": {"type": "number"},
                    },
                    "required": ["title", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_context_window",
                "description": "Summarize the most recent durable context records in the active thread.",
                "parameters": {
                    "type": "object",
                    "properties": {"top_k": {"type": "integer"}},
                },
            },
        },
    ]


def context_tools_for_anthropic() -> list[dict[str, Any]]:
    if not _enabled():
        return []
    tools = []
    for openai_tool in context_tools_for_openai():
        function = openai_tool["function"]
        tools.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    return tools


def merge_tools(user_tools: Iterable[Any] | None, *, anthropic: bool = False) -> list[Any]:
    merged = list(user_tools or [])
    wrapper_tools = context_tools_for_anthropic() if anthropic else context_tools_for_openai()
    user_names = set()
    for tool in merged:
        if isinstance(tool, dict):
            if anthropic:
                name = tool.get("name")
            else:
                function = tool.get("function", {})
                name = function.get("name")
            if name:
                user_names.add(name)

    for tool in wrapper_tools:
        name = tool["name"] if anthropic else tool["function"]["name"]
        if name not in user_names:
            merged.append(tool)
    return merged


def execute_context_tool(name: str, arguments_json: str, thread_id: str) -> str:
    store = context_store()
    if name == "search_context":
        args = SearchContextArgs.model_validate_json(arguments_json)
        results = store.search(
            thread_id=thread_id,
            query=args.query,
            kinds=args.kinds,
            file_paths=args.file_paths,
            symbols=args.symbols,
            top_k=min(args.top_k, _search_top_k_max()),
        )
        return json.dumps(
            {
                "results": [
                    {
                        "memory_id": item["record"].memory_id,
                        "kind": item["record"].kind,
                        "title": item["record"].title,
                        "summary": _truncate(item["record"].summary, _summary_chars()),
                        "file_paths": item["record"].file_paths[:2],
                        "symbol_names": item["record"].symbol_names[:4],
                        "score": round(item["score"], 4),
                    }
                    for item in results
                ]
            },
            ensure_ascii=False,
        )

    if name == "get_context_by_ids":
        args = GetContextByIdsArgs.model_validate_json(arguments_json)
        records = store.get_by_ids(thread_id, args.ids[: _get_by_ids_max_records()])
        return json.dumps(
            {
                "records": [
                    {
                        "memory_id": record.memory_id,
                        "kind": record.kind,
                        "title": record.title,
                        "summary": _truncate(record.summary, _summary_chars()),
                        "content": _truncate(record.content, _record_content_chars()),
                        "file_paths": record.file_paths[:2],
                        "symbol_names": record.symbol_names[:4],
                        "metadata": record.metadata,
                    }
                    for record in records
                ]
            },
            ensure_ascii=False,
        )

    if name == "get_thread_state":
        args = GetThreadStateArgs.model_validate_json(arguments_json or "{}")
        snapshot = store.get_hot_snapshot(thread_id)
        if snapshot is None:
            return json.dumps({"thread_id": thread_id, "state": None}, ensure_ascii=False)
        payload = snapshot.model_dump(mode="json")
        if not args.include_recent_messages:
            payload["recent_messages"] = []
        else:
            payload["recent_messages"] = [
                _truncate(item, _summary_chars())
                for item in payload.get("recent_messages", [])[: _thread_state_recent_messages_max()]
            ]
        for key in ("current_objective",):
            if payload.get(key):
                payload[key] = _truncate(payload[key], _record_content_chars())
        for key in ("current_plan", "unresolved_blockers", "recent_decisions", "latest_tool_failures"):
            payload[key] = [_truncate(item, _summary_chars()) for item in payload.get(key, [])[:4]]
        return json.dumps({"thread_id": thread_id, "state": payload}, ensure_ascii=False)

    if name == "store_context_note":
        args = StoreContextNoteArgs.model_validate_json(arguments_json)
        record = store.new_memory(
            thread_id=thread_id,
            kind=args.kind,
            title=args.title,
            content=args.content,
            summary=args.content[:220],
            file_paths=args.file_paths,
            symbol_names=args.symbol_names,
            importance=args.importance,
            recency_score=0.8,
            metadata={"source": "tool"},
        )
        store.upsert_memories([record])
        return json.dumps({"stored": True, "memory_id": record.memory_id}, ensure_ascii=False)

    if name == "summarize_context_window":
        args = SummarizeContextWindowArgs.model_validate_json(arguments_json or "{}")
        records = store.recent(thread_id, min(args.top_k, _search_top_k_max()))
        return json.dumps(
            {
                "summary": [
                    {
                        "memory_id": record.memory_id,
                        "kind": record.kind,
                        "title": record.title,
                        "summary": _truncate(record.summary, _summary_chars()),
                    }
                    for record in records
                ]
            },
            ensure_ascii=False,
        )

    raise ValueError(f"Unknown context tool: {name}")


def context_tool_instruction() -> str:
    if not _enabled():
        return ""
    return (
        "Wrapper context tools are available as optional durable memory retrieval helpers. "
        "Use them only when prior goals, tool failures, file facts, decisions, or older thread context may materially help the answer. "
        "Treat `search_context` as semantic RAG-style lookup, then use `get_context_by_ids` or `get_thread_state` if you need exact records or the current working snapshot. "
        "If the current prompt already contains enough information, answer directly without calling these tools. "
        "Do not combine wrapper context tools with external user tools in the same response; fetch context first, then continue."
    )
