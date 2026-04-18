import pytest

from app.services.staged_ingestion import prepare_staged_messages
from app.services.staged_ingestion_store import reset_ingestion_progress


@pytest.mark.asyncio
async def test_prepare_staged_messages_ingests_large_latest_user_message(monkeypatch):
    reset_ingestion_progress()
    monkeypatch.setenv("TUBS_ENABLE_STAGED_INGESTION", "true")
    monkeypatch.setenv("TUBS_STAGED_INGEST_BACKEND", "memory")
    monkeypatch.setenv("TUBS_STAGED_INGEST_THRESHOLD_TOKENS", "10")
    monkeypatch.setenv("TUBS_STAGED_INGEST_BLOCK_TOKENS", "4")
    monkeypatch.setenv("TUBS_APPROX_CHARS_PER_TOKEN", "4")

    sent_payloads = []

    async def fake_send(payload, images, bearer_token, stream):
        sent_payloads.append(payload)
        return {
            "type": "done",
            "response": "ACK",
            "thread": {"id": "thread_ingest"},
        }

    monkeypatch.setattr("app.services.staged_ingestion.async_send_tubs_request", fake_send)

    messages = [
        {"role": "user", "content": "Short setup"},
        {
            "role": "user",
            "content": (
                "First paragraph explains the repo problem in detail. "
                "Second paragraph includes the failing stack trace summary. "
                "Third paragraph includes a code diff and the final task to fix it."
            ),
        },
    ]

    result = await prepare_staged_messages(
        model="gpt-5.4",
        messages=messages,
        thread_id=None,
        conversation_key="conv_1",
        bearer_token="token",
    )

    assert result.applied is True
    assert result.thread_id == "thread_ingest"
    assert len(sent_payloads) >= 2
    assert "Context block 1/" in sent_payloads[0]["prompt"]
    assert "pre-ingested into this thread" in result.messages[-1]["content"]


@pytest.mark.asyncio
async def test_prepare_staged_messages_reuses_completed_ingestion(monkeypatch):
    reset_ingestion_progress()
    monkeypatch.setenv("TUBS_ENABLE_STAGED_INGESTION", "true")
    monkeypatch.setenv("TUBS_STAGED_INGEST_BACKEND", "memory")
    monkeypatch.setenv("TUBS_STAGED_INGEST_THRESHOLD_TOKENS", "10")
    monkeypatch.setenv("TUBS_STAGED_INGEST_BLOCK_TOKENS", "4")
    monkeypatch.setenv("TUBS_APPROX_CHARS_PER_TOKEN", "4")

    call_count = {"value": 0}

    async def fake_send(payload, images, bearer_token, stream):
        call_count["value"] += 1
        return {
            "type": "done",
            "response": "ACK",
            "thread": {"id": "thread_ingest"},
        }

    monkeypatch.setattr("app.services.staged_ingestion.async_send_tubs_request", fake_send)

    messages = [
        {
            "role": "user",
            "content": (
                "Large block one with implementation details. "
                "Large block two with more implementation details. "
                "Large block three with final task details."
            ),
        }
    ]

    first = await prepare_staged_messages(
        model="gpt-5.4",
        messages=messages,
        thread_id=None,
        conversation_key="conv_reuse",
        bearer_token="token",
    )
    first_call_count = call_count["value"]

    second = await prepare_staged_messages(
        model="gpt-5.4",
        messages=messages,
        thread_id=first.thread_id,
        conversation_key="conv_reuse",
        bearer_token="token",
    )

    assert first.applied is True
    assert second.applied is True
    assert call_count["value"] == first_call_count
