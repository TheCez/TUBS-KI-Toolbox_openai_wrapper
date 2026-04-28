import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import HTTPException

from app.main import app


async def _collect_sse_chunks(response) -> list[str]:
    body = []
    async for chunk in response.aiter_text():
        body.append(chunk)
    return body


@pytest.mark.asyncio
async def test_responses_non_streaming(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert payload["model"] == "gpt-5.4"
        assert "Speak plainly." in payload["customInstructions"]
        assert bearer_token == "test-token"
        assert stream is False
        assert images == []
        return {
            "type": "done",
            "response": "Hello from TU-BS",
            "promptTokens": 11,
            "responseTokens": 7,
            "totalTokens": 18,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "instructions": "Speak plainly.",
                "store": False,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["model"] == "gpt-5.4"
    assert data["output_text"] == "Hello from TU-BS"
    assert data["usage"]["input_tokens"] == 11
    assert data["usage"]["output_tokens"] == 7
    assert data["output"][0]["content"][0]["type"] == "output_text"


@pytest.mark.asyncio
async def test_responses_accepts_shorthand_input_item(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert payload["prompt"] == "[User]: Hello from shorthand"
        return {
            "type": "done",
            "response": "Hello back",
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": [{"content": "Hello from shorthand"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["output_text"] == "Hello back"


@pytest.mark.asyncio
async def test_responses_accepts_lobechat_tool_loop_items(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[Tool Result]: <searchResults>done</searchResults>" in payload["prompt"]
        return {
            "type": "done",
            "response": "Summarized answer",
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": [
                    {"content": "Hi what is the latest situation?", "role": "user"},
                    {
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "lobe-web-browsing____search____builtin",
                        "arguments": "{\"query\":\"latest US Iran conflict news developments\"}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_123",
                        "output": "<searchResults>done</searchResults>",
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["output_text"] == "Summarized answer"


@pytest.mark.asyncio
async def test_responses_adds_repair_hint_for_string_replace_errors(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Wrapper repair hint:" in payload["prompt"]
        assert "smaller anchored replacements" in payload["prompt"]
        assert "target file: `C:\\personal\\portfolio\\cowork_test_project\\src\\components\\PrimaryButton.tsx`" in payload["prompt"]
        return {
            "type": "done",
            "response": "Retry after rereading.",
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": (
                                    "Update(C:\\personal\\portfolio\\cowork_test_project\\src\\components\\PrimaryButton.tsx)\n"
                                    "Error: String to replace not found in file.\n"
                                    "String: const baseClassName =\n"
                                    "  'inline-flex items-center justify-center'"
                                ),
                                "is_error": True,
                            }
                        ],
                    },
                    {"role": "user", "content": "Try again."},
                ],
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_responses_accepts_content_blocks_with_extra_fields(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Use the configured skills." in payload["customInstructions"]
        assert "[User]: Hello from block list" in payload["prompt"]
        return {
            "type": "done",
            "response": "Processed blocks",
            "promptTokens": 6,
            "responseTokens": 3,
            "totalTokens": 9,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "text",
                                "text": "Use the configured skills.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Hello from block list", "annotations": []}
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["output_text"] == "Processed blocks"


@pytest.mark.asyncio
async def test_responses_streaming(monkeypatch):
    async def fake_stream():
        yield {"type": "chunk", "content": "Hello "}
        yield {"type": "chunk", "content": "world"}
        yield {
            "type": "done",
            "response": "Hello world",
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert payload["model"] == "gpt-5.4"
        assert stream is True
        return fake_stream()

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with ac.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={"model": "gpt-5.4", "input": "Hi", "stream": True},
        ) as response:
            chunks = await _collect_sse_chunks(response)

    joined = "".join(chunks)
    assert response.status_code == 200
    assert "event: response.created" in joined
    assert "event: response.output_text.delta" in joined
    assert "Hello " in joined
    assert "world" in joined
    assert "event: response.completed" in joined


@pytest.mark.asyncio
async def test_responses_streaming_hides_tool_xml_and_emits_function_call(monkeypatch):
    async def fake_stream():
        yield {
            "type": "chunk",
            "content": '<tool_calls><tool_call><name>search</name><arguments>{"query":"latest news"}</arguments></tool_call></tool_calls>',
        }
        yield {
            "type": "done",
            "response": '<tool_calls><tool_call><name>search</name><arguments>{"query":"latest news"}</arguments></tool_call></tool_calls>Here is a summary',
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        return fake_stream()

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with ac.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": "Hi",
                "stream": True,
                "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            },
        ) as response:
            chunks = await _collect_sse_chunks(response)

    joined = "".join(chunks)
    assert response.status_code == 200
    assert "<tool_calls>" not in joined
    assert '"type": "function_call"' in joined
    assert '"name": "search"' in joined


@pytest.mark.asyncio
async def test_responses_reasoning_is_added_to_custom_instructions(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        instructions = payload["customInstructions"]
        assert "Reason thoroughly. Prefer correctness over brevity." in instructions
        assert "Target at most about 300 tokens." in instructions
        return {
            "type": "done",
            "response": "Detailed answer",
            "promptTokens": 11,
            "responseTokens": 7,
            "totalTokens": 18,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": "Explain quantum entanglement",
                "reasoning": {"effort": "high"},
                "max_output_tokens": 300,
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_responses_downgrades_invalid_tool_call_to_text(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        return {
            "type": "done",
            "response": '<tool_calls><tool_call><name>AskUserQuestion</name><arguments>{}</arguments></tool_call></tool_calls>',
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": "Ask me something",
                "tools": [
                    {
                        "type": "function",
                        "name": "AskUserQuestion",
                        "parameters": {
                            "type": "object",
                            "properties": {"questions": {"type": "array"}},
                            "required": ["questions"],
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_text"]
    assert "missing required fields: questions" in data["output_text"]
    assert not any(item["type"] == "function_call" for item in data["output"])


@pytest.mark.asyncio
async def test_responses_rotates_exhausted_tubs_thread(monkeypatch):
    monkeypatch.setattr("app.api.routes.responses.get_cached_thread_id", lambda _key: "thread_exhausted")
    calls = {"count": 0}

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        calls["count"] += 1
        if calls["count"] == 1:
            assert payload.get("thread") == "thread_exhausted"
            raise HTTPException(status_code=429, detail="Sie haben das Token limit für dieses Gespräch überschritten.")
        assert payload.get("thread") is None
        return {
            "type": "done",
            "response": "Recovered response",
            "promptTokens": 5,
            "responseTokens": 3,
            "totalTokens": 8,
            "thread": {"id": "thread_fresh"},
        }

    monkeypatch.setattr("app.api.routes.responses.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "input": "Please continue the task",
            },
        )

    assert response.status_code == 200
    assert response.json()["output_text"] == "Recovered response"
    assert calls["count"] == 2
