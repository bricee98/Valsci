"""Base provider abstraction for LLM gateway providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.llm.types import ProviderRequest, ProviderResponse


class BaseProvider(ABC):
    provider_name: str = "unknown"
    base_url: str | None = None

    @abstractmethod
    async def chat(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

