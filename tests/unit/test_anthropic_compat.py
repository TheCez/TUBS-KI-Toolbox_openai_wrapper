import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


async def _collect_sse_chunks(response) -> list[str]:
    chunks = []
    async for chunk in response.aiter_text():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_anthropic_accepts_compat_fields(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Do not call any tools in this response." in payload["customInstructions"]
        assert "Reason carefully before answering and provide enough detail to fully address the request." in payload["customInstructions"]
        return {
            "type": "done",
            "response": "Plain answer",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-sonnet-4-0",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"name": "search", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "none"},
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "metadata": {"source": "claude-code"},
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Plain answer"


@pytest.mark.asyncio
async def test_anthropic_tool_instructions_include_required_argument_summary(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        instructions = payload["customInstructions"]
        assert "Tool requirements summary:" in instructions
        assert "- AskUserQuestion: Ask the user a question" in instructions
        assert "Required arguments: questions" in instructions
        assert "Never call a tool with an empty or incomplete arguments object" in instructions
        return {
            "type": "done",
            "response": "Plain answer",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-sonnet-4-0",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [
                    {
                        "name": "AskUserQuestion",
                        "description": "Ask the user a question",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                            "required": ["questions"],
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_anthropic_accepts_context_management(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        return {
            "type": "done",
            "response": "Plain answer",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
        }

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-sonnet-4-0",
                "messages": [{"role": "user", "content": "Hello"}],
                "context_management": {
                    "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Plain answer"


@pytest.mark.asyncio
async def test_anthropic_prompt_includes_tool_history(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[Tool Intention]:" in payload["prompt"]
        assert "[Tool Result OK id=toolu_1]: sunny" in payload["prompt"]
        return {
            "type": "done",
            "response": "Final answer",
            "promptTokens": 6,
            "responseTokens": 3,
            "totalTokens": 9,
        }

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-sonnet-4-0",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"city": "Berlin"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_anthropic_prompt_marks_tool_result_errors(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[Tool Result ERROR id=toolu_1]: Error writing file" in payload["prompt"]
        return {
            "type": "done",
            "response": "I could not write the file",
            "promptTokens": 6,
            "responseTokens": 3,
            "totalTokens": 9,
        }

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-sonnet-4-0",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "write_file",
                                "input": {"path": "src/app.ts", "content": "hello"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Error writing file",
                                "is_error": True,
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "I could not write the file"


@pytest.mark.asyncio
async def test_anthropic_accepts_block_extras_and_function_call_output_shapes(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[User]: Hello from block content" in payload["prompt"]
        assert "[Tool Intention]: search({\"query\": \"latest\"}) [id=call_1]" in payload["prompt"]
        assert "[Tool Result OK id=call_1]: done" in payload["prompt"]
        return {
            "type": "done",
            "response": "Summarized",
            "promptTokens": 9,
            "responseTokens": 4,
            "totalTokens": 13,
        }

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-opus-4-1",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello from block content",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "search",
                                "arguments": {"query": "latest"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "function_call_output",
                                "call_id": "call_1",
                                "output": "done",
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Summarized"


@pytest.mark.asyncio
async def test_anthropic_streaming_hides_tool_xml(monkeypatch):
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

    monkeypatch.setattr("app.api.routes.anthropic.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with ac.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "test-token"},
            json={
                "model": "claude-sonnet-4-0",
                "messages": [{"role": "user", "content": "Latest news?"}],
                "stream": True,
                "tools": [{"name": "search", "input_schema": {"type": "object"}}],
            },
        ) as response:
            chunks = await _collect_sse_chunks(response)

    joined = "".join(chunks)
    assert response.status_code == 200
    assert "<tool_calls>" not in joined
    assert '"type": "tool_use"' in joined
