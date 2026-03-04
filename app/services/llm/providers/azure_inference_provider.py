"""Azure AI Inference provider adapter."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from app.services.llm.providers.base import BaseProvider
from app.services.llm.types import ProviderRequest, ProviderResponse


class AzureInferenceProvider(BaseProvider):
    provider_name = "azure-inference"

    def __init__(self, endpoint: str, api_key: str):
        self.base_url = endpoint
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
        except Exception as exc:  # pragma: no cover - import depends on optional package
            raise RuntimeError("azure-ai-inference package is required for azure-inference provider") from exc

        self._system_message_type = None
        self._user_message_type = None
        try:
            from azure.ai.inference.models import SystemMessage, UserMessage

            self._system_message_type = SystemMessage
            self._user_message_type = UserMessage
        except Exception:
            self._system_message_type = None
            self._user_message_type = None

        self.client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    async def chat(self, request: ProviderRequest) -> ProviderResponse:
        messages = self._build_messages(request.messages)
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": request.model,
            "max_tokens": int(request.max_output_tokens or 1024),
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        response = await asyncio.to_thread(self.client.complete, **kwargs)
        raw_text = response.choices[0].message.content or ""
        # Azure AI Inference currently does not consistently provide usage counts.
        input_tokens = max(1, int(sum(len(str(m.get("content", ""))) for m in request.messages) / 4))
        output_tokens = max(1, int(len(raw_text) / 4))
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "is_estimated": True,
        }
        return ProviderResponse(
            raw_text=raw_text,
            model_used=request.model,
            finish_reason=getattr(response.choices[0], "finish_reason", None),
            usage=usage,
            raw_response=response.as_dict() if hasattr(response, "as_dict") else response,
        )

    def _build_messages(self, messages):
        if self._system_message_type is None or self._user_message_type is None:
            return messages
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                converted.append(self._system_message_type(content=content))
            else:
                converted.append(self._user_message_type(content=content))
        return converted

