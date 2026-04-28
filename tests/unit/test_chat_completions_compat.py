import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.conversation_state import reset_thread_cache


async def _collect_sse_chunks(response) -> list[str]:
    chunks = []
    async for chunk in response.aiter_text():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_chat_completions_accepts_extra_compat_fields(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Target at most about 120 tokens." in payload["customInstructions"]
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
async def test_chat_completions_accepts_block_content_with_extra_fields(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Follow the harness rules." in payload["customInstructions"]
        assert "[User]: Hello from Claude Code" in payload["prompt"]
        return {
            "type": "done",
            "response": "Handled block content",
            "promptTokens": 10,
            "responseTokens": 4,
            "totalTokens": 14,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "text",
                                "text": "Follow the harness rules.",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello from Claude Code",
                                "citations": [],
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Handled block content"


@pytest.mark.asyncio
async def test_chat_completions_accepts_tool_blocks_inside_content(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[Assistant Tool Calls]: get_weather({\"city\": \"Berlin\"}) [id=toolu_1]" in payload["prompt"]
        assert "[Tool Result]: sunny" in payload["prompt"]
        return {
            "type": "done",
            "response": "Weather summary",
            "promptTokens": 8,
            "responseTokens": 4,
            "totalTokens": 12,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
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
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Weather summary"


@pytest.mark.asyncio
async def test_chat_completions_adds_repair_hint_for_string_replace_errors(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "Wrapper repair hint:" in payload["prompt"]
        assert "Read the file again before editing" in payload["prompt"]
        assert "smaller anchored replacements" in payload["prompt"]
        assert "target file: `C:\\personal\\portfolio\\cowork_test_project\\src\\components\\PrimaryButton.tsx`" in payload["prompt"]
        assert "likely stable anchor: `const baseClassName`" in payload["prompt"]
        return {
            "type": "done",
            "response": "Try a smaller anchored edit.",
            "promptTokens": 8,
            "responseTokens": 4,
            "totalTokens": 12,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [
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
async def test_chat_completions_adds_completion_hint_for_successful_file_write(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        assert "[Wrapper Completion Hint]:" in payload["prompt"]
        assert "mark the related task or todo as completed" in payload["prompt"]
        assert "successful file operation on `fibonacci.py`" in payload["prompt"]
        return {
            "type": "done",
            "response": "The file is created and the task can be closed.",
            "promptTokens": 8,
            "responseTokens": 4,
            "totalTokens": 12,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Wrote 12 lines to fibonacci.py",
                                "is_error": False,
                            }
                        ],
                    },
                    {"role": "user", "content": "Close out the task if the requested file is done."},
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


@pytest.mark.asyncio
async def test_chat_completions_downgrades_invalid_tool_call_to_text(monkeypatch):
    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        return {
            "type": "done",
            "response": '<tool_calls><tool_call><name>AskUserQuestion</name><arguments>{}</arguments></tool_call></tool_calls>',
            "promptTokens": 3,
            "responseTokens": 2,
            "totalTokens": 5,
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Ask me something"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "AskUserQuestion",
                            "parameters": {
                                "type": "object",
                                "properties": {"questions": {"type": "array"}},
                                "required": ["questions"],
                            },
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["tool_calls"] is None
    assert "missing required fields: questions" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_chat_completions_reuses_tubs_thread_and_compacts_history(monkeypatch):
    reset_thread_cache()
    monkeypatch.setenv("TUBS_KEEP_LAST_TURNS", "4")
    monkeypatch.setenv("TUBS_COMPACT_SUMMARY_CHARS", "1000")
    monkeypatch.setenv("TUBS_THREAD_PROMPT_TOKENS", "40")
    monkeypatch.setenv("TUBS_THREAD_SUMMARY_CHARS", "120")

    call_count = {"value": 0}

    async def fake_send_tubs_request(payload, images, bearer_token, stream):
        call_count["value"] += 1
        if call_count["value"] == 1:
            assert payload.get("thread") is None
            return {
                "type": "done",
                "response": "First answer",
                "promptTokens": 4,
                "responseTokens": 2,
                "totalTokens": 6,
                "thread": {"id": "thread_123"},
            }

        assert payload.get("thread") == "thread_123"
        assert "[User]: Latest request" in payload["prompt"]
        assert "[User]: Initial request" not in payload["prompt"]
        return {
            "type": "done",
            "response": "Second answer",
            "promptTokens": 4,
            "responseTokens": 2,
            "totalTokens": 6,
            "thread": {"id": "thread_123"},
        }

    monkeypatch.setattr("app.api.routes.chat.async_send_tubs_request", fake_send_tubs_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        first = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Initial request " + ("alpha " * 20)}],
            },
        )
        second = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "model": "gpt-5.4",
                "messages": [
                    {"role": "user", "content": "Initial request " + ("alpha " * 20)},
                    {"role": "assistant", "content": "Initial answer " + ("beta " * 16)},
                    {"role": "user", "content": "Follow-up request " + ("gamma " * 16)},
                    {"role": "assistant", "content": "Follow-up answer " + ("delta " * 16)},
                    {"role": "user", "content": "Latest request"},
                ],
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
