from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping

from app.services.context_store import context_store


def _csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _user_agent(headers: Mapping[str, str] | None) -> str:
    if not headers:
        return ""
    for key, value in headers.items():
        if key.lower() == "user-agent":
            return value.lower()
    return ""


@dataclass(frozen=True)
class ThreadPolicy:
    use_upstream_threads: bool
    reuse_upstream_thread: bool
    strict_wrapper_state: bool
    minimal_upstream_mode: bool
    client_name: str


def resolve_thread_policy(*, endpoint: str, headers: Mapping[str, str] | None = None) -> ThreadPolicy:
    user_agent = _user_agent(headers)
    client_name = "generic"
    if "openclaw" in user_agent:
        client_name = "openclaw"
    elif "claude" in user_agent:
        client_name = "claude-code"
    elif "lobechat" in user_agent:
        client_name = "lobechat"

    use_upstream_threads = os.getenv("TUBS_USE_UPSTREAM_THREADS", "true").strip().lower() == "true"
    strict_wrapper_state = os.getenv("TUBS_STRICT_WRAPPER_STATE_MODE", "false").strip().lower() == "true"
    reuse_upstream_thread = True
    openclaw_strict_default = os.getenv("TUBS_OPENCLAW_STRICT_WRAPPER_STATE", "true").strip().lower() == "true"
    minimal_upstream_mode = False
    openclaw_minimal_default = os.getenv("TUBS_OPENCLAW_MINIMAL_MODE", "true").strip().lower() == "true"
    claude_code_strict_default = os.getenv("TUBS_CLAUDE_CODE_STRICT_WRAPPER_STATE", "true").strip().lower() == "true"
    claude_code_minimal_default = os.getenv("TUBS_CLAUDE_CODE_MINIMAL_MODE", "true").strip().lower() == "true"

    no_thread_clients = _csv_env("TUBS_NO_UPSTREAM_THREAD_CLIENTS")
    strict_clients = _csv_env("TUBS_STRICT_WRAPPER_STATE_CLIENTS")
    no_thread_endpoints = _csv_env("TUBS_NO_UPSTREAM_THREAD_ENDPOINTS")

    if client_name in no_thread_clients or endpoint.lower() in no_thread_endpoints:
        reuse_upstream_thread = False
        use_upstream_threads = False
    if client_name == "openclaw" and openclaw_strict_default:
        strict_wrapper_state = True
        use_upstream_threads = False
        reuse_upstream_thread = False
    if client_name == "openclaw" and openclaw_minimal_default:
        minimal_upstream_mode = True
    if client_name == "claude-code" and claude_code_strict_default:
        strict_wrapper_state = True
        use_upstream_threads = False
        reuse_upstream_thread = False
    if client_name == "claude-code" and claude_code_minimal_default:
        minimal_upstream_mode = True
    if client_name in strict_clients:
        strict_wrapper_state = True
        use_upstream_threads = False
        reuse_upstream_thread = False

    return ThreadPolicy(
        use_upstream_threads=use_upstream_threads,
        reuse_upstream_thread=reuse_upstream_thread,
        strict_wrapper_state=strict_wrapper_state,
        minimal_upstream_mode=minimal_upstream_mode,
        client_name=client_name,
    )


def policy_allows_upstream_thread(*, thread_id: str, policy: ThreadPolicy) -> bool:
    if not policy.reuse_upstream_thread:
        return False
    snapshot = context_store().get_hot_snapshot(thread_id)
    if snapshot is None:
        return True
    disabled_until = snapshot.thread_control.upstream_threads_disabled_until
    if disabled_until and disabled_until > datetime.now(UTC):
        return False
    return True
