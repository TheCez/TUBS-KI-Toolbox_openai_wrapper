from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ContextMemoryKind = Literal[
    "goal",
    "constraint",
    "decision",
    "plan_step",
    "tool_failure",
    "file_fact",
    "code_summary",
    "user_request",
    "open_question",
    "assistant_response",
    "context_note",
]


class ContextMemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    memory_id: str
    kind: ContextMemoryKind
    title: str
    content: str
    summary: str
    source_turn_range: str | None = None
    source_tool: str | None = None
    file_paths: list[str] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    importance: float = 0.5
    recency_score: float = 0.5
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class SearchContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    kinds: list[str] | None = None
    file_paths: list[str] | None = None
    symbols: list[str] | None = None
    top_k: int = 5


class GetContextByIdsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str]


class GetThreadStateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_recent_messages: bool = True


class StoreContextNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    content: str
    kind: ContextMemoryKind = "context_note"
    file_paths: list[str] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    importance: float = 0.6


class SummarizeContextWindowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = 8


class HotContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    current_objective: str | None = None
    current_plan: list[str] = Field(default_factory=list)
    unresolved_blockers: list[str] = Field(default_factory=list)
    recent_decisions: list[str] = Field(default_factory=list)
    latest_tool_failures: list[str] = Field(default_factory=list)
    recent_messages: list[str] = Field(default_factory=list)
    updated_at: datetime
