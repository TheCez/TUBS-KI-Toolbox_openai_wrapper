import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import HTTPException
from app.models.tubs import is_local_model

_BASE = os.getenv("TUBS_BASE_URL", "https://ki-toolbox.tu-braunschweig.de")
TUBS_CLOUD_URL = f"{_BASE}/api/v1/chat/send"
TUBS_LOCAL_URL = f"{_BASE}/api/v1/localChat/send"


class RequestGate:
    def __init__(self, max_concurrent_requests: int, min_interval_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))
        self._spacing_lock = asyncio.Lock()
        self._min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_started_at = 0.0
        self._cooldown_until = 0.0

    @asynccontextmanager
    async def slot(self):
        async with self._semaphore:
            await self._wait_for_turn()
            yield

    async def _wait_for_turn(self) -> None:
        async with self._spacing_lock:
            now = time.perf_counter()
            wait_seconds = max(0.0, self._cooldown_until - now)
            if self._min_interval_seconds > 0:
                wait_seconds = max(wait_seconds, self._min_interval_seconds - (now - self._last_started_at))
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_started_at = time.perf_counter()

    def note_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        cooldown = retry_after_seconds
        if cooldown is None:
            cooldown = float(os.getenv("TUBS_RATE_LIMIT_COOLDOWN_SECONDS", "8"))
        cooldown = max(0.0, cooldown)
        self._cooldown_until = max(self._cooldown_until, time.perf_counter() + cooldown)


_REQUEST_GATE = RequestGate(
    max_concurrent_requests=int(os.getenv("TUBS_MAX_CONCURRENT_REQUESTS", "1")),
    min_interval_seconds=float(os.getenv("TUBS_MIN_REQUEST_INTERVAL_SECONDS", "0")),
)

def _raise_http_exception(status_code: int, body: str) -> None:
    detail = body
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            detail = parsed.get("message") or parsed.get("error") or body
    except json.JSONDecodeError:
        pass
    if status_code == 429:
        _REQUEST_GATE.note_rate_limit()
    raise HTTPException(status_code=status_code, detail=detail)


def _load_ndjson_line(line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid NDJSON chunk from TU-BS backend: {line[:200]}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Unexpected non-object NDJSON chunk from TU-BS backend")
    return payload


async def _non_stream_response(client: httpx.AsyncClient, url: str, headers: dict, req_kwargs: dict):
    try:
        async with _REQUEST_GATE.slot():
            response = await client.post(url, headers=headers, **req_kwargs)
            if response.status_code != 200:
                _raise_http_exception(response.status_code, response.text)

            # KI-Toolbox API streams NDJSON by default, we capture it all and wait for "done" chunk
            final_data = {}
            for line in response.iter_lines():
                if line:
                    chunk = _load_ndjson_line(line)
                    if chunk.get("type") == "done":
                        final_data = chunk
                        break
            if not final_data:
                raise HTTPException(status_code=502, detail="TU-BS backend did not return a terminal done chunk")
            return final_data
    finally:
        await client.aclose()

async def _stream_response(client: httpx.AsyncClient, url: str, headers: dict, req_kwargs: dict):
    try:
        async with _REQUEST_GATE.slot():
            async with client.stream("POST", url, headers=headers, **req_kwargs) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    _raise_http_exception(response.status_code, body)

                async for line in response.aiter_lines():
                    if line:
                        yield _load_ndjson_line(line)
    finally:
        await client.aclose()

async def async_send_tubs_request(
    payload: dict,
    images: list,
    bearer_token: str,
    stream: bool
):
    url = TUBS_LOCAL_URL if is_local_model(payload.get("model", "")) else TUBS_CLOUD_URL
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }
    
    req_kwargs = {}
    if images:
        files = []
        for img in images:
            fname, fbytes, ftype = img
            files.append(("chatAttachment", (fname, fbytes, ftype)))
        
        data = {
            "jsonBody": json.dumps(payload)
        }
        req_kwargs["data"] = data
        req_kwargs["files"] = files
    else:
        headers["Content-Type"] = "application/json"
        req_kwargs["json"] = payload
        
    client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0))
    
    if stream:
        return _stream_response(client, url, headers, req_kwargs)
    else:
        return await _non_stream_response(client, url, headers, req_kwargs)
