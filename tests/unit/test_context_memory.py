import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.context_ingest import context_ingest_service
from app.services.context_store import reset_context_store
from app.services.context_tools import context_tool_instruction, context_tools_for_openai, execute_context_tool
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
    assert "optional durable memory retrieval helpers" in context_tool_instruction().lower()


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
