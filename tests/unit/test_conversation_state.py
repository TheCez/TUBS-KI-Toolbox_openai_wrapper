from app.services.conversation_state import (
    build_conversation_key,
    compact_messages,
    get_cached_thread_id,
    remember_thread_id,
    reset_thread_cache,
)


def test_compact_messages_keeps_recent_turns_and_summarizes_older(monkeypatch):
    monkeypatch.setenv("TUBS_KEEP_LAST_TURNS", "2")
    monkeypatch.setenv("TUBS_COMPACT_SUMMARY_CHARS", "1000")

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First request with lots of detail."},
        {"role": "assistant", "content": "First reply."},
        {"role": "user", "content": "Second request."},
        {"role": "assistant", "content": "Second reply."},
        {"role": "user", "content": "Latest request."},
    ]

    recent, summary = compact_messages(messages)

    assert summary is not None
    assert "Earlier conversation summary:" in summary
    assert "First request with lots of detail." in summary
    assert len(recent) == 3
    assert recent[0]["role"] == "system"
    assert recent[-1]["content"] == "Latest request."


def test_thread_cache_remembers_thread_by_conversation_key():
    reset_thread_cache()
    conversation_key = build_conversation_key(
        bearer_token="token-123",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Build me a site"}],
    )

    assert get_cached_thread_id(conversation_key) is None
    remember_thread_id(conversation_key, {"thread": {"id": "thread_abc"}})
    assert get_cached_thread_id(conversation_key) == "thread_abc"
