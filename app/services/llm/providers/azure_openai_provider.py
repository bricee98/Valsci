"""Azure OpenAI provider using OpenAI SDK compatible client."""

from __future__ import annotations

from typing import Any, Dict

try:
    import openai
except Exception:  # pragma: no cover - optional dependency for tests/local contexts
    openai = None

from app.services.llm.providers.base import BaseProvider
from app.services.llm.types import ProviderRequest, ProviderResponse


class AzureOpenAIProvider(BaseProvider):
    provider_name = "azure-openai"

    def __init__(self, api_key: str, endpoint: str, api_version: str):
        if openai is None:
            raise RuntimeError("openai package is required for Azure OpenAI provider.")
        self.base_url = endpoint
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    async def chat(self, request: ProviderRequest) -> ProviderResponse:
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }
        if request.response_format:
            kwargs["response_format"] = request.response_format
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = int(request.max_output_tokens)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.timeout_seconds is not None:
            kwargs["timeout"] = int(request.timeout_seconds)
        if request.extra_headers:
            kwargs["extra_headers"] = request.extra_headers

        response = await self.async_client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        raw_text = message.content or ""
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
                "is_estimated": False,
            }
        return ProviderResponse(
            raw_text=raw_text,
            model_used=getattr(response, "model", request.model),
            finish_reason=getattr(response.choices[0], "finish_reason", None),
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
        )
