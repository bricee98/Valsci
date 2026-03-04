"""Async concurrency + RPM/TPM limiter for LLM calls."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Tuple


class GatewayRateLimiter:
    def __init__(self, max_concurrency: int, requests_per_minute: int, tokens_per_minute: int):
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.requests_per_minute = max(0, int(requests_per_minute or 0))
        self.tokens_per_minute = max(0, int(tokens_per_minute or 0))
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()
        self._request_events: Deque[float] = deque()
        self._token_events: Deque[Tuple[float, int]] = deque()

    @asynccontextmanager
    async def reserve(self, estimated_tokens: int):
        await self._semaphore.acquire()
        try:
            await self._wait_for_window(estimated_tokens=max(1, int(estimated_tokens or 1)))
            yield
        finally:
            self._semaphore.release()

    async def adjust_usage(self, estimated_tokens: int, actual_tokens: int) -> None:
        delta = int(actual_tokens or 0) - int(estimated_tokens or 0)
        if delta <= 0:
            return
        async with self._lock:
            now = time.time()
            self._prune_old_events(now)
            self._token_events.append((now, delta))

    async def _wait_for_window(self, estimated_tokens: int) -> None:
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.time()
                self._prune_old_events(now)
                req_count = len(self._request_events)
                token_count = sum(tokens for _, tokens in self._token_events)

                req_ok = self.requests_per_minute == 0 or req_count < self.requests_per_minute
                token_ok = self.tokens_per_minute == 0 or (token_count + estimated_tokens) <= self.tokens_per_minute

                if req_ok and token_ok:
                    self._request_events.append(now)
                    self._token_events.append((now, estimated_tokens))
                    return

                if self.requests_per_minute and req_count >= self.requests_per_minute:
                    wait_time = max(wait_time, 60.0 - (now - self._request_events[0]) + 0.01)
                if self.tokens_per_minute and (token_count + estimated_tokens) > self.tokens_per_minute:
                    wait_time = max(wait_time, self._token_wait_time(now, token_count + estimated_tokens))

            await asyncio.sleep(max(wait_time, 0.05))

    def _token_wait_time(self, now: float, desired_total: int) -> float:
        if self.tokens_per_minute == 0:
            return 0.01
        running_total = 0
        for timestamp, tokens in self._token_events:
            running_total += tokens
            if (desired_total - running_total) <= self.tokens_per_minute:
                return max(0.01, 60.0 - (now - timestamp) + 0.01)
        return 1.0

    def _prune_old_events(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_events and self._request_events[0] < cutoff:
            self._request_events.popleft()
        while self._token_events and self._token_events[0][0] < cutoff:
            self._token_events.popleft()

