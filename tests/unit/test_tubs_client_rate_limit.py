import asyncio
import time

import pytest

from app.services.tubs_client import RequestGate


@pytest.mark.asyncio
async def test_request_gate_limits_concurrency():
    gate = RequestGate(max_concurrent_requests=1, min_interval_seconds=0)
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        async with gate.slot():
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1

    await asyncio.gather(worker(), worker(), worker())

    assert max_active == 1


@pytest.mark.asyncio
async def test_request_gate_enforces_minimum_spacing():
    gate = RequestGate(max_concurrent_requests=2, min_interval_seconds=0.05)
    started = []

    async def worker():
        async with gate.slot():
            started.append(time.perf_counter())
            await asyncio.sleep(0.01)

    await asyncio.gather(worker(), worker())

    assert len(started) == 2
    assert started[1] - started[0] >= 0.045
