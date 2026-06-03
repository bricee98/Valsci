"""Ollama provider adapter using OpenAI-compatible endpoint for chat and native show API for metadata."""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from app.services.llm.providers.openai_provider import OpenAICompatibleProvider


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def ollama_openai_base_url(base_url: str | None) -> str:
    raw = str(base_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if not raw:
        raw = DEFAULT_OLLAMA_BASE_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    lower_raw = raw.lower()
    if lower_raw.endswith("/v1"):
        return raw
    if lower_raw.endswith("/api"):
        return f"{raw[:-4]}/v1"
    return f"{raw}/v1"


class OllamaProvider(OpenAICompatibleProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str | None):
        super().__init__(api_key="sk-no-key-required", base_url=ollama_openai_base_url(base_url))

    async def fetch_model_details(self, model_name: str, show_url: str, timeout_seconds: int = 10) -> Optional[Dict]:
        payload = {"model": model_name}
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(show_url, json=payload)
            response.raise_for_status()
            return response.json()
