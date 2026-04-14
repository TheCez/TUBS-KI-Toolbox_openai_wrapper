import pytest
from httpx import ASGITransport, AsyncClient

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
