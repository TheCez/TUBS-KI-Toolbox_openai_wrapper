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


class GetPinnedStateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetPinnedStateFieldArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: Literal[
        "user_name",
        "assistant_name",
        "assistant_creature",
        "assistant_vibe",
        "assistant_emoji",
        "bootstrap_status",
        "bootstrap_expected_reply",
        "workflow_kind",
        "workflow_status",
    ]
    value: str


class MarkWorkflowCompleteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str | None = None


class GetDebugTraceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = 20


class PinnedUserIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None


class PinnedAssistantIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    creature: str | None = None
    vibe: str | None = None
    emoji: str | None = None


class BootstrapState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["unknown", "pending", "answered", "completed"] = "unknown"
    pending_step: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    answered_fields: list[str] = Field(default_factory=list)
    last_exact_expected_reply: str | None = None


class ActiveWorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str | None = None
    status: str | None = None
    current_goal: str | None = None
    blocked_on_user: bool = False


class TaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open_tasks: list[str] = Field(default_factory=list)
    in_progress_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)


class ThreadControlState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upstream_thread_id: str | None = None
    rotation_count: int = 0
    low_information_reply_count: int = 0
    poisoned_thread_count: int = 0
    upstream_threads_disabled_until: datetime | None = None
    last_good_answer_at: datetime | None = None
    last_bad_filler_at: datetime | None = None


class CompactionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    source_turn_range: str | None = None
    kind: str = "bridge"
    summary: str
    workflow_kind: str | None = None
    workflow_status: str | None = None
    created_at: datetime


class ProtectedWorkingSetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["file_read", "tool_output", "agent_summary", "plan_summary"] = "file_read"
    title: str
    file_path: str | None = None
    content: str
    source_tool: str | None = None
    updated_at: datetime


class HotContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    user_identity: PinnedUserIdentity = Field(default_factory=PinnedUserIdentity)
    assistant_identity: PinnedAssistantIdentity = Field(default_factory=PinnedAssistantIdentity)
    bootstrap_state: BootstrapState = Field(default_factory=BootstrapState)
    active_workflow: ActiveWorkflowState = Field(default_factory=ActiveWorkflowState)
    task_state: TaskState = Field(default_factory=TaskState)
    thread_control: ThreadControlState = Field(default_factory=ThreadControlState)
    compaction_artifacts: list[CompactionArtifact] = Field(default_factory=list)
    protected_working_set: list[ProtectedWorkingSetEntry] = Field(default_factory=list)
    hidden_bridge_summary: str | None = None
    current_objective: str | None = None
    current_plan: list[str] = Field(default_factory=list)
    unresolved_blockers: list[str] = Field(default_factory=list)
    recent_decisions: list[str] = Field(default_factory=list)
    latest_tool_failures: list[str] = Field(default_factory=list)
    recent_messages: list[str] = Field(default_factory=list)
    updated_at: datetime
