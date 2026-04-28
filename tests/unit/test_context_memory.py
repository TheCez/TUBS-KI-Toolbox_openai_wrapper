import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.openai import Message
from app.services.context_ingest import context_ingest_service
from app.services.context_store import reset_context_store
from app.services.context_tools import context_tool_instruction, context_tools_for_openai, execute_context_tool
from app.services.openai_bridge import build_tubs_payload_from_messages
from app.services.conversation_state import build_conversation_key, reset_thread_cache


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
