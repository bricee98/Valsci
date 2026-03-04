"""llama.cpp OpenAI-compatible provider adapter."""

from __future__ import annotations

from app.services.llm.providers.openai_provider import OpenAICompatibleProvider


class LlamaCppProvider(OpenAICompatibleProvider):
    provider_name = "llamacpp"

    def __init__(self, base_url: str):
        super().__init__(api_key="sk-no-key-required", base_url=base_url)

