"""Adaptive (AIMD) concurrency limiter for model-provider calls."""

from __future__ import annotations

import asyncio

from app.adapters.openai_compat import AdaptiveLimiter


def test_halves_on_rate_limit_and_recovers_additively():
    lim = AdaptiveLimiter(max_limit=32)
    lim.on_rate_limited()
    assert lim.limit == 16
    lim.on_rate_limited()
    assert lim.limit == 8
    for _ in range(8):
        lim.on_success()
    assert lim.limit == 9  # +1 after `limit` consecutive successes
    for _ in range(10):
        lim.on_rate_limited()
    assert lim.limit == 2  # never below the floor


async def test_never_exceeds_limit_under_load():
    lim = AdaptiveLimiter(max_limit=4)
    peak = 0

    async def worker() -> None:
        nonlocal peak
        async with lim:
            peak = max(peak, lim.in_flight)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(worker() for _ in range(40)))
    assert peak == 4 and lim.in_flight == 0
