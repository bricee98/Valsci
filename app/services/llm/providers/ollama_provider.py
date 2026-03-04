"""Ollama provider adapter using OpenAI-compatible endpoint for chat and native show API for metadata."""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from app.services.llm.providers.openai_provider import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str):
        super().__init__(api_key="sk-no-key-required", base_url=base_url)

    async def fetch_model_details(self, model_name: str, show_url: str, timeout_seconds: int = 10) -> Optional[Dict]:
        payload = {"model": model_name}
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(show_url, json=payload)
            response.raise_for_status()
            return response.json()

