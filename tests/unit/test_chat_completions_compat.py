import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


async def _collect_sse_chunks(response) -> list[str]:
    chunks = []
    async for chunk in response.aiter_text():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_chat_completions_accepts_extra_compat_fields(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Keep the final answer within roughly 120 tokens" in payload["customInstructions"]
        assert "Do not call any tools in this response." in payload["customInstructions"]
        return {
            "type": "done",
            "response": "Plain answer",
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
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 120,
                "store": False,
                "metadata": {"source": "lobechat"},
                "parallel_tool_calls": True,
                "service_tier": "default",
                "stream_options": {"include_usage": True},
                "modalities": ["text"],
                "tool_choice": "none",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Plain answer"


@pytest.mark.asyncio
async def test_chat_completions_prompt_includes_tool_history(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[Assistant Tool Calls]:" in payload["prompt"]
        assert "[Tool Result]: sunny" in payload["prompt"]
        return {
            "type": "done",
            "response": "Final answer",
            "promptTokens": 6,
            "responseTokens": 3,
            "totalTokens": 9,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ],
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_streaming_hides_tool_xml(monkeypatch):
    async def fake_stream():
        yield {
            "type": "chunk",
            "content": '<tool_calls><tool_call><name>search</name><arguments>{"query":"latest"}</arguments></tool_call></tool_calls>',
        }
        yield {
            "type": "done",
            "response": '<tool_calls><tool_call><name>search</name><arguments>{"query":"latest"}</arguments></tool_call></tool_calls>Here is prose',
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        return fake_stream()

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with ac.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Latest news?"}],
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
            },
        ) as response:
            chunks = await _collect_sse_chunks(response)

    joined = "".join(chunks)
    assert response.status_code == 200
    assert "<tool_calls>" not in joined
    assert '"tool_calls"' in joined
    assert '"finish_reason": "tool_calls"' in joined
