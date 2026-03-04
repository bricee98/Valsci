"""Retry strategy for transient LLM/provider failures."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import asyncio
import httpx


@dataclass
class RetryDecision:
    should_retry: bool
    reason: str


class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        backoff_jitter: float = 0.5,
    ):
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_seconds = max(0.1, float(backoff_base_seconds))
        self.backoff_max_seconds = max(self.backoff_base_seconds, float(backoff_max_seconds))
        self.backoff_jitter = max(0.0, float(backoff_jitter))

    def classify(self, exc: Exception, http_status: Optional[int] = None) -> RetryDecision:
        status = http_status or self._extract_http_status(exc)

        if status in {429, 500, 502, 503, 504}:
            return RetryDecision(True, f"http_{status}")
        if status == 400:
            return RetryDecision(False, "bad_request")

        transient_types = (
            TimeoutError,
            asyncio.TimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
        )
        if isinstance(exc, transient_types):
            return RetryDecision(True, exc.__class__.__name__)

        openai_status = getattr(exc, "status_code", None)
        if openai_status in {429, 500, 502, 503, 504}:
            return RetryDecision(True, f"http_{openai_status}")

        return RetryDecision(False, exc.__class__.__name__)

    def compute_backoff_seconds(self, attempt_number: int) -> float:
        exponent = max(0, attempt_number - 1)
        delay = min(self.backoff_base_seconds * (2 ** exponent), self.backoff_max_seconds)
        if self.backoff_jitter > 0:
            delay += random.uniform(0, self.backoff_jitter)
        return delay

    @staticmethod
    def _extract_http_status(exc: Exception) -> Optional[int]:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        if response is not None:
            code = getattr(response, "status_code", None)
            if isinstance(code, int):
                return code
        return None

