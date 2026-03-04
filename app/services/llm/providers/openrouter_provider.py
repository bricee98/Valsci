"""OpenRouter provider (OpenAI-compatible API)."""

from __future__ import annotations

from typing import Dict, Optional

from app.services.llm.providers.openai_provider import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    provider_name = "openrouter"

    def __init__(self, api_key: str, base_url: str, extra_headers: Optional[Dict[str, str]] = None):
        super().__init__(api_key=api_key, base_url=base_url, extra_headers=extra_headers)

