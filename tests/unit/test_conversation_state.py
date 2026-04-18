from app.services.conversation_state import (
    build_prompt_with_compaction,
    build_conversation_key,
    compact_messages,
    estimate_token_count,
    get_cached_thread_id,
    remember_thread_id,
    reset_thread_cache,
)
import app.services.conversation_state as conversation_state


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


def test_thread_cache_uses_redis_backend_when_configured(monkeypatch):
    class FakeRedisModule:
        @staticmethod
        def from_url(*args, **kwargs):
            return fake_client

    class FakeRedisClient:
        def __init__(self):
            self.store = {}

        def ping(self):
            return True

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, ex=None):
            self.store[key] = value

        def keys(self, pattern):
            prefix = pattern[:-1]
            return [key for key in self.store if key.startswith(prefix)]

        def delete(self, *keys):
            for key in keys:
                self.store.pop(key, None)

    fake_client = FakeRedisClient()

    monkeypatch.setenv("TUBS_THREAD_CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/0")
    monkeypatch.setattr(conversation_state, "Redis", FakeRedisModule)

    reset_thread_cache()
    conversation_key = build_conversation_key(
        bearer_token="token-456",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Build me a dashboard"}],
    )

    remember_thread_id(conversation_key, {"thread": {"id": "thread_xyz"}})
    assert get_cached_thread_id(conversation_key) == "thread_xyz"


def test_build_prompt_with_compaction_uses_thread_budget_and_keeps_latest_request(monkeypatch):
    monkeypatch.setenv("TUBS_KEEP_LAST_TURNS", "4")
    monkeypatch.setenv("TUBS_MAX_PROMPT_TOKENS", "200")
    monkeypatch.setenv("TUBS_THREAD_PROMPT_TOKENS", "40")
    monkeypatch.setenv("TUBS_THREAD_SUMMARY_CHARS", "120")

    messages = [
        {"role": "user", "content": "First request " + ("alpha " * 30)},
        {"role": "assistant", "content": "First answer " + ("beta " * 20)},
        {"role": "user", "content": "Second request " + ("gamma " * 20)},
        {"role": "assistant", "content": "Second answer " + ("delta " * 20)},
        {"role": "user", "content": "Latest request with the exact task we must preserve."},
    ]

    prompt = build_prompt_with_compaction(
        messages,
        compile_prompt=lambda items: "\n".join(
            f"[{item['role'].capitalize()}]: {item['content']}" for item in items if item["role"] != "system"
        ),
        thread_id="thread_123",
    )

    assert "Latest request with the exact task we must preserve." in prompt
    assert estimate_token_count(prompt) <= 40
    assert "First request alpha" not in prompt


def test_build_prompt_with_compaction_summarizes_older_messages_without_thread(monkeypatch):
    monkeypatch.setenv("TUBS_KEEP_LAST_TURNS", "2")
    monkeypatch.setenv("TUBS_MAX_PROMPT_TOKENS", "80")
    monkeypatch.setenv("TUBS_COMPACT_SUMMARY_CHARS", "220")

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Old request with setup."},
        {"role": "assistant", "content": "Old reply."},
        {"role": "user", "content": "Newest request."},
    ]

    prompt = build_prompt_with_compaction(
        messages,
        compile_prompt=lambda items: "\n".join(
            f"[{item['role'].capitalize()}]: {item['content']}" for item in items if item["role"] != "system"
        ),
        thread_id=None,
    )

    assert "Earlier conversation summary:" in prompt
    assert "Newest request." in prompt


def test_build_prompt_with_compaction_iteratively_folds_old_blocks(monkeypatch):
    monkeypatch.setenv("TUBS_KEEP_LAST_TURNS", "2")
    monkeypatch.setenv("TUBS_MAX_PROMPT_TOKENS", "70")
    monkeypatch.setenv("TUBS_THREAD_PROMPT_TOKENS", "70")
    monkeypatch.setenv("TUBS_THREAD_SUMMARY_CHARS", "260")

    messages = []
    for idx in range(1, 9):
        messages.append({"role": "user", "content": f"Request {idx} " + ("detail " * 18)})
        messages.append({"role": "assistant", "content": f"Reply {idx} " + ("response " * 16)})
    messages.append({"role": "user", "content": "Final request that must remain intact."})

    prompt = build_prompt_with_compaction(
        messages,
        compile_prompt=lambda items: "\n".join(
            f"[{item['role'].capitalize()}]: {item['content']}" for item in items if item["role"] != "system"
        ),
        thread_id="thread_rollup",
    )

    assert "Final request that must remain intact." in prompt
    assert estimate_token_count(prompt) <= 70
    assert "Request 1 detail" not in prompt
