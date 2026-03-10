"""OpenAI and OpenAI-compatible provider implementations."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import openai
except Exception:  # pragma: no cover - optional dependency for tests/local contexts
    openai = None

from app.services.llm.providers.base import BaseProvider
from app.services.llm.types import ProviderRequest, ProviderResponse


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self, api_key: str, base_url: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None):
        if openai is None:
            raise RuntimeError("openai package is required for OpenAI-compatible providers.")
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        if base_url:
            self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.async_client = openai.AsyncOpenAI(api_key=api_key)

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
        elif self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

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
            http_status=None,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
        )


class OpenAICompatibleProvider(OpenAIProvider):
    provider_name = "openai_compat"
