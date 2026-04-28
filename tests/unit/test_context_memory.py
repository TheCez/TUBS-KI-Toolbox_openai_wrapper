import json
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.openai import Message
from app.services.context_ingest import context_ingest_service
from app.services.context_runtime import fresh_thread_rehydration_instruction, protected_working_set_instruction
from app.services.context_store import reset_context_store
from app.services.context_tools import context_tool_instruction, context_tools_for_openai, execute_context_tool
from app.services.context_runtime import pinned_state_instruction
from app.services.openai_bridge import build_tubs_payload_from_messages
from app.services.conversation_state import build_conversation_key, reset_thread_cache
from app.services.debug_trace import record_debug_event
from app.services.thread_policy import ThreadPolicy, policy_allows_upstream_thread, resolve_thread_policy
from app.services.thread_recovery import reset_upstream_thread_state


@pytest.fixture(autouse=True)
def reset_context_state():
    reset_context_store()
    reset_thread_cache()
    yield
    reset_context_store()
    reset_thread_cache()


def test_context_tools_search_and_get_state():
    service = context_ingest_service()
    thread_id = "thread-alpha"
    service.ingest_turn(
        thread_id,
        [
            {
                "role": "user",
                "content": "I want to fix PrimaryButton.tsx and keep the rounded button style.",
            },
            {
                "role": "tool",
                "content": "Error writing file in C:\\personal\\portfolio\\cowork_test_project\\src\\components\\PrimaryButton.tsx",
            },
        ],
        response_text="We should reread PrimaryButton.tsx before editing.",
    )

    search_payload = json.loads(
        execute_context_tool(
            "search_context",
            json.dumps({"query": "PrimaryButton file write failure", "top_k": 3}),
            thread_id,
        )
    )
    assert search_payload["results"]
    assert any("PrimaryButton.tsx" in json.dumps(item) for item in search_payload["results"])

    state_payload = json.loads(
        execute_context_tool(
            "get_thread_state",
            json.dumps({"include_recent_messages": True}),
            thread_id,
        )
    )
    assert state_payload["state"]["current_objective"].startswith("I want to fix PrimaryButton.tsx")
    assert state_payload["state"]["recent_messages"]


def test_pinned_state_extracts_identity_and_bootstrap_completion():
    service = context_ingest_service()
    thread_id = "thread-bootstrap"
    service.ingest_turn(
        thread_id,
        [
            {
                "role": "user",
                "content": (
                    "I’m mid-bootstrap and waiting on you.\n"
                    "Pick anything you like for these four:\n"
                    "• my name\n• what kind of creature I am\n• my vibe\n• my emoji\n"
                    "Reply:\nyes, call me Ajay"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Name : Jarvis\n"
                    "Creature : Friendly and fun helper\n"
                    "Vibe: cool fun and helpful\n"
                    "Emoji: 😎\n"
                    "My name : Ajay"
                ),
            },
        ],
        response_text="Bootstrap acknowledged.",
    )

    state_payload = json.loads(
        execute_context_tool(
            "get_thread_state",
            json.dumps({"include_recent_messages": False}),
            thread_id,
        )
    )
    state = state_payload["state"]
    assert state["user_identity"]["name"] == "Ajay"
    assert state["assistant_identity"]["name"] == "Jarvis"
    assert state["assistant_identity"]["creature"] == "Friendly and fun helper"
    assert state["bootstrap_state"]["status"] == "completed"
    instruction = pinned_state_instruction(thread_id) or ""
    assert "Assistant name: Jarvis" in instruction
    assert "Do not ask bootstrap identity questions again" in instruction


def test_recent_file_reads_are_preserved_in_protected_working_set():
    service = context_ingest_service()
    thread_id = "thread-working-set"
    service.ingest_turn(
        thread_id,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read",
                        "content": (
                            "Read(C:\\personal\\portfolio\\cowork_test_project\\src\\components\\SiteShell.tsx)\n"
                            "Read 12 lines\n"
                            "1 import { Outlet } from 'react-router-dom'\n"
                            "2 import SiteHeader from './SiteHeader'\n"
                            "3\n"
                            "4 export default function SiteShell() {\n"
                            "5   return <main><Outlet /></main>\n"
                            "6 }\n"
                        ),
                        "is_error": False,
                    }
                ],
            }
        ],
        response_text="I will update the shell next.",
    )

    state_payload = json.loads(
        execute_context_tool(
            "get_thread_state",
            json.dumps({"include_recent_messages": False}),
            thread_id,
        )
    )
    working_set = state_payload["state"]["protected_working_set"]
    assert working_set
    assert working_set[0]["file_path"].endswith("SiteShell.tsx")

    instruction = protected_working_set_instruction(thread_id) or ""
    assert "Protected working set:" in instruction
    assert "SiteShell.tsx" in instruction
    assert "import { Outlet } from 'react-router-dom'" in instruction


def test_structured_plan_outputs_are_preserved_in_protected_working_set():
    service = context_ingest_service()
    thread_id = "thread-plan-working-set"
    response_text = (
        "Lowest-risk refresh strategy\n\n"
        "- keep Tailwind as-is\n"
        "- add shared animation/background CSS in src/index.css\n"
        "- apply shell-level layout polish in src/components/SiteShell.tsx\n"
        "- refresh homepage presentation in src/pages/HomePage.tsx\n\n"
        "Bottom line\n\n"
        "- global polish: src/index.css\n"
        "- app shell/background: src/components/SiteShell.tsx\n"
        "- homepage wow-factor: src/pages/HomePage.tsx\n"
    )
    service.ingest_turn(
        thread_id,
        [{"role": "user", "content": "Plan the animated background refresh."}],
        response_text=response_text,
    )

    state_payload = json.loads(
        execute_context_tool(
            "get_thread_state",
            json.dumps({"include_recent_messages": False}),
            thread_id,
        )
    )
    working_set = state_payload["state"]["protected_working_set"]
    assert working_set
    assert any(entry["kind"] == "plan_summary" for entry in working_set)

    instruction = protected_working_set_instruction(thread_id) or ""
    assert "Protected working set:" in instruction
    assert "Lowest-risk refresh strategy" in instruction
    assert "src/components/SiteShell.tsx" in instruction


def test_context_tool_metadata_marks_tools_as_optional_rag_helpers():
    search_tool = next(
        tool for tool in context_tools_for_openai() if tool["function"]["name"] == "search_context"
    )
    assert "optional" in search_tool["function"]["description"].lower()
    assert "rag" in search_tool["function"]["description"].lower()
    assert "optional wrapper memory tools" in context_tool_instruction().lower()


def test_get_context_by_ids_is_bounded_for_prompt_safety(monkeypatch):
    monkeypatch.setenv("TUBS_CONTEXT_GET_BY_IDS_MAX_RECORDS", "1")
    monkeypatch.setenv("TUBS_CONTEXT_RECORD_CONTENT_CHARS", "120")
    service = context_ingest_service()
    thread_id = "thread-bounded"
    long_text = "Important context. " + ("detail " * 200)
    service.ingest_turn(
        thread_id,
        [{"role": "user", "content": long_text}],
        response_text="Stored.",
    )
    search_payload = json.loads(
        execute_context_tool(
            "search_context",
            json.dumps({"query": "important context", "top_k": 2}),
            thread_id,
        )
    )
    first_id = search_payload["results"][0]["memory_id"]
    details_payload = json.loads(
        execute_context_tool(
            "get_context_by_ids",
            json.dumps({"ids": [first_id]}),
            thread_id,
        )
    )
    assert len(details_payload["records"]) == 1
    assert len(details_payload["records"][0]["content"]) <= 120


def test_pinned_state_tools_and_debug_trace_round_trip():
    service = context_ingest_service()
    thread_id = "thread-tools"
    service.ingest_turn(
        thread_id,
        [{"role": "user", "content": "Name : Jarvis\nMy name : Ajay"}],
        response_text="Stored.",
    )
    updated = json.loads(
        execute_context_tool(
            "set_pinned_state_field",
            json.dumps({"field": "workflow_kind", "value": "bootstrap"}),
            thread_id,
        )
    )
    assert updated["updated"] is True
    completed = json.loads(
        execute_context_tool(
            "mark_workflow_complete",
            json.dumps({"summary": "Bootstrap finished cleanly."}),
            thread_id,
        )
    )
    assert completed["workflow_status"] == "completed"
    pinned = json.loads(execute_context_tool("get_pinned_state", "{}", thread_id))
    assert pinned["pinned_state"]["user_identity"]["name"] == "Ajay"
    assert pinned["pinned_state"]["assistant_identity"]["name"] == "Jarvis"
    assert pinned["pinned_state"]["active_workflow"]["status"] == "completed"

    record_debug_event(thread_id, "test_event", {"ok": True})
    trace = json.loads(execute_context_tool("get_debug_trace", json.dumps({"limit": 5}), thread_id))
    assert trace["events"]
    assert trace["events"][-1]["event"] == "test_event"


def test_thread_policy_disables_upstream_threads_when_poisoned():
    service = context_ingest_service()
    thread_id = "thread-poisoned"
    service.ingest_turn(thread_id, [{"role": "user", "content": "hello"}], response_text="hi")
    from app.services.context_store import context_store

    snapshot = context_store().get_hot_snapshot(thread_id)
    assert snapshot is not None
    snapshot.thread_control.upstream_threads_disabled_until = datetime.now(UTC) + timedelta(minutes=5)
    context_store().set_hot_snapshot(snapshot)

    assert policy_allows_upstream_thread(
        thread_id=thread_id,
        policy=ThreadPolicy(use_upstream_threads=True, reuse_upstream_thread=True, strict_wrapper_state=False, minimal_upstream_mode=False, client_name="generic"),
    ) is False


def test_reset_upstream_thread_state_increments_rotation_count():
    service = context_ingest_service()
    thread_id = "thread-rotate"
    service.ingest_turn(thread_id, [{"role": "user", "content": "hello"}], response_text="hi")
    from app.services.context_store import context_store

    snapshot = context_store().get_hot_snapshot(thread_id)
    assert snapshot is not None
    snapshot.thread_control.upstream_thread_id = "thread_old"
    context_store().set_hot_snapshot(snapshot)

    reset_upstream_thread_state(thread_id)
    snapshot = context_store().get_hot_snapshot(thread_id)
    assert snapshot is not None
    assert snapshot.thread_control.rotation_count == 1
    assert snapshot.thread_control.upstream_thread_id is None


def test_maintenance_prompt_is_ignored_for_hot_context():
    service = context_ingest_service()
    thread_id = "thread-maintenance"
    service.ingest_turn(
        thread_id,
        [
            {
                "role": "user",
                "content": (
                    "Pre-compaction memory flush. Store durable memories only in memory/2026-04-28.md. "
                    "Treat workspace bootstrap/reference files such as MEMORY.md, DREAMS.md, SOUL.md, TOOLS.md, "
                    "and AGENTS.md as read-only during this flush; reply with NO_REPLY."
                ),
            }
        ],
        response_text="NO_REPLY",
    )
    from app.services.context_store import context_store

    snapshot = context_store().get_hot_snapshot(thread_id)
    assert snapshot is not None
    assert snapshot.current_objective is None
    assert snapshot.hidden_bridge_summary is None
    recent = context_store().recent(thread_id, 5)
    assert not recent


def test_openclaw_defaults_to_strict_wrapper_state(monkeypatch):
    monkeypatch.delenv("TUBS_OPENCLAW_STRICT_WRAPPER_STATE", raising=False)
    policy = resolve_thread_policy(endpoint="anthropic", headers={"User-Agent": "OpenClaw/1.0"})
    assert policy.client_name == "openclaw"
    assert policy.strict_wrapper_state is True
    assert policy.use_upstream_threads is False
    assert policy.reuse_upstream_thread is False


@pytest.mark.asyncio
async def test_chat_completions_resolves_wrapper_context_tool(monkeypatch):
    service = context_ingest_service()
    explicit_user = "ctx-thread-1"
    message = {"role": "user", "content": "How should I update SectionCard.tsx?"}
    conversation_key = build_conversation_key(
        bearer_token="test-token",
        model="gpt-5.4",
        messages=[message],
        explicit_user=explicit_user,
    )
    service.ingest_turn(
        conversation_key,
        [{"role": "user", "content": "We already decided to use anchored edits in SectionCard.tsx."}],
        response_text="Use anchored edits around SectionCard rather than giant replacements.",
    )

    calls = {"count": 0}

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        calls["count"] += 1
        assert stream is False
        if calls["count"] == 1:
            assert "search_context" in payload["customInstructions"]
            return {
                "type": "done",
                "response": (
                    '<tool_calls><tool_call><name>search_context</name>'
                    '<arguments>{"query":"What did we decide about SectionCard.tsx?","top_k":2}</arguments>'
                    "</tool_call></tool_calls>"
                ),
                "promptTokens": 5,
                "responseTokens": 3,
                "totalTokens": 8,
                "thread": {"id": "thread_abc"},
            }

        assert '"kind": "assistant_response"' in payload["prompt"] or "anchored edits" in payload["prompt"]
        return {
            "type": "done",
            "response": "Use the earlier anchored edit decision for SectionCard.tsx.",
            "promptTokens": 7,
            "responseTokens": 4,
            "totalTokens": 11,
            "thread": {"id": "thread_abc"},
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "user": explicit_user,
                "messages": [message],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Use the earlier anchored edit decision for SectionCard.tsx."
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_chat_completions_skips_wrapper_context_tools_without_stored_state(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "search_context" not in (payload.get("customInstructions") or "")
        return {
            "type": "done",
            "response": "Fresh answer",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Hello on a fresh thread"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Fresh answer"


@pytest.mark.asyncio
async def test_chat_completions_includes_pinned_state_instruction(monkeypatch):
    service = context_ingest_service()
    explicit_user = "thread-pinned-route"
    conversation_key = build_conversation_key(
        bearer_token="test-token",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Hi"}],
        explicit_user=explicit_user,
    )
    service.ingest_turn(
        conversation_key,
        [
            {
                "role": "user",
                "content": "Name : Jarvis\nCreature : helpful machine familiar\nVibe: warm and sharp\nEmoji: 😎\nMy name : Ajay",
            }
        ],
        response_text="Identity stored.",
    )

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Pinned thread state:" in payload["customInstructions"]
        assert "User name: Ajay" in payload["customInstructions"]
        assert "Assistant name: Jarvis" in payload["customInstructions"]
        return {
            "type": "done",
            "response": "Pinned answer",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "user": explicit_user,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Pinned answer"


@pytest.mark.asyncio
async def test_chat_completions_includes_protected_working_set_instruction(monkeypatch):
    service = context_ingest_service()
    explicit_user = "thread-working-set-route"
    conversation_key = build_conversation_key(
        bearer_token="test-token",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Continue editing the shell"}],
        explicit_user=explicit_user,
    )
    service.ingest_turn(
        conversation_key,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read",
                        "content": (
                            "Read(C:\\personal\\portfolio\\cowork_test_project\\src\\components\\SiteShell.tsx)\n"
                            "Read 6 lines\n"
                            "1 import { Outlet } from 'react-router-dom'\n"
                            "2 import SiteHeader from './SiteHeader'\n"
                            "3 export default function SiteShell() {\n"
                            "4   return <main><Outlet /></main>\n"
                            "5 }\n"
                        ),
                        "is_error": False,
                    }
                ],
            }
        ],
        response_text="Use the current shell file contents for the next edit.",
    )

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Protected working set:" in payload["customInstructions"]
        assert "SiteShell.tsx" in payload["customInstructions"]
        assert "import { Outlet } from 'react-router-dom'" in payload["customInstructions"]
        return {
            "type": "done",
            "response": "Working set preserved.",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "user": explicit_user,
                "messages": [{"role": "user", "content": "Continue editing the shell"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Working set preserved."


@pytest.mark.asyncio
async def test_chat_completions_includes_recent_plan_working_set_instruction(monkeypatch):
    service = context_ingest_service()
    explicit_user = "thread-plan-working-set-route"
    conversation_key = build_conversation_key(
        bearer_token="test-token",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Continue the UI polish implementation"}],
        explicit_user=explicit_user,
    )
    service.ingest_turn(
        conversation_key,
        [{"role": "user", "content": "Explore the current UI structure and styling."}],
        response_text=(
            "Lowest-risk refresh strategy\n\n"
            "- keep Tailwind as-is\n"
            "- add shared animation/background CSS in src/index.css\n"
            "- apply shell-level layout polish in src/components/SiteShell.tsx\n"
            "- refresh homepage presentation in src/pages/HomePage.tsx\n"
        ),
    )

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Protected working set:" in payload["customInstructions"]
        assert "Lowest-risk refresh strategy" in payload["customInstructions"]
        assert "src/pages/HomePage.tsx" in payload["customInstructions"]
        return {
            "type": "done",
            "response": "Plan carried forward.",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "user": explicit_user,
                "messages": [{"role": "user", "content": "Continue the UI polish implementation"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Plan carried forward."


@pytest.mark.asyncio
async def test_chat_completions_adds_fresh_thread_rehydration_instruction(monkeypatch):
    service = context_ingest_service()
    explicit_user = "thread-rehydrate-route"
    conversation_key = build_conversation_key(
        bearer_token="test-token",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Please continue"}],
        explicit_user=explicit_user,
    )
    service.ingest_turn(
        conversation_key,
        [{"role": "user", "content": "We were implementing the popup calculator."}],
        response_text="Keep the popup calculator plan in mind.",
    )
    assert fresh_thread_rehydration_instruction(conversation_key)

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Fresh thread rehydration:" in payload["customInstructions"]
        assert "popup calculator" in payload["customInstructions"].lower()
        return {
            "type": "done",
            "response": "Continuing with the existing plan.",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "user": explicit_user,
                "messages": [{"role": "user", "content": "Please continue"}],
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_can_disable_upstream_threads_for_client(monkeypatch):
    monkeypatch.setenv("TUBS_USE_UPSTREAM_THREADS", "true")
    monkeypatch.setenv("TUBS_NO_UPSTREAM_THREAD_CLIENTS", "openclaw")
    monkeypatch.setattr("app.api.routes.chat.get_cached_thread_id", lambda _key: "thread_123")

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert payload.get("thread") is None
        return {
            "type": "done",
            "response": "Fresh thread only.",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-token",
                "User-Agent": "OpenClaw/1.0",
            },
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Continue"}],
            },
        )

    assert response.status_code == 200


def test_prompt_budget_accounts_for_large_instruction_overhead(monkeypatch):
    monkeypatch.setenv("TUBS_MAX_PROMPT_TOKENS", "120")
    monkeypatch.setenv("TUBS_INSTRUCTION_TOKEN_RESERVE", "20")
    messages = [
        Message(role="user", content="Earlier request " + ("alpha " * 50)),
        Message(role="assistant", content="Earlier answer " + ("beta " * 40)),
        Message(role="user", content="Latest request that must survive."),
    ]
    payload, _images, _model = build_tubs_payload_from_messages(
        model="gpt-5.4",
        messages=messages,
        instructions="System guidance. " + ("rules " * 180),
    )
    assert "Latest request that must survive." in payload["prompt"]
    assert "Earlier request alpha" not in payload["prompt"]


@pytest.mark.asyncio
async def test_chat_completions_overflow_requires_context_retrieval_before_final_answer(monkeypatch):
    monkeypatch.setenv("TUBS_ENABLE_STAGED_INGESTION", "false")
    monkeypatch.setenv("TUBS_MAX_PROMPT_TOKENS", "80")
    monkeypatch.setenv("TUBS_INSTRUCTION_TOKEN_RESERVE", "10")
    monkeypatch.setenv("TUBS_CONTEXT_TOOL_LOOP_LIMIT", "4")

    calls = {"count": 0}

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        calls["count"] += 1
        assert stream is False
        if calls["count"] == 1:
            assert "overflow context mode" in payload["customInstructions"]
            return {
                "type": "done",
                "response": "I already know the answer.",
                "promptTokens": 5,
                "responseTokens": 3,
                "totalTokens": 8,
                "thread": {"id": "thread_overflow"},
            }

        if calls["count"] == 2:
            assert "overflow context mode" in payload["customInstructions"]
            assert "Before any final answer or external tool call" in payload["customInstructions"]
            return {
                "type": "done",
                "response": (
                    '<tool_calls><tool_call><name>search_context</name>'
                    '<arguments>{"query":"popup calculator integration","top_k":2}</arguments>'
                    "</tool_call></tool_calls>"
                ),
                "promptTokens": 6,
                "responseTokens": 4,
                "totalTokens": 10,
                "thread": {"id": "thread_overflow"},
            }

        assert "popup calculator integration" in payload["prompt"]
        return {
            "type": "done",
            "response": "Implement the popup calculator using the retrieved thread context.",
            "promptTokens": 7,
            "responseTokens": 4,
            "totalTokens": 11,
            "thread": {"id": "thread_overflow"},
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    long_request = "Please implement popup calculator integration. " + ("details " * 500)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": long_request}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Implement the popup calculator using the retrieved thread context."
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_chat_completions_overflow_retries_after_low_information_final(monkeypatch):
    monkeypatch.setenv("TUBS_ENABLE_STAGED_INGESTION", "false")
    monkeypatch.setenv("TUBS_MAX_PROMPT_TOKENS", "80")
    monkeypatch.setenv("TUBS_INSTRUCTION_TOKEN_RESERVE", "10")
    monkeypatch.setenv("TUBS_CONTEXT_TOOL_LOOP_LIMIT", "5")

    calls = {"count": 0}

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "type": "done",
                "response": (
                    '<tool_calls><tool_call><name>search_context</name>'
                    '<arguments>{"query":"bootstrap identity", "top_k":2}</arguments>'
                    "</tool_call></tool_calls>"
                ),
                "promptTokens": 5,
                "responseTokens": 3,
                "totalTokens": 8,
                "thread": {"id": "thread_boot"},
            }
        if calls["count"] == 2:
            return {
                "type": "done",
                "response": "Nothing else to say here",
                "promptTokens": 5,
                "responseTokens": 3,
                "totalTokens": 8,
                "thread": {"id": "thread_boot"},
            }
        assert "previous reply did not answer" in (payload.get("customInstructions") or "")
        assert "Do not reply with a placeholder" in (payload.get("customInstructions") or "")
        return {
            "type": "done",
            "response": "yes, call me Ajay",
            "promptTokens": 6,
            "responseTokens": 4,
            "totalTokens": 10,
            "thread": {"id": "thread_boot"},
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    long_request = "Please finish bootstrap and stop forgetting who I am. " + ("identity " * 500)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": long_request}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "yes, call me Ajay"
    assert calls["count"] == 3
