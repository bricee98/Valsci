"""Per-run LLM gateway cache keyed by provider snapshot."""

from __future__ import annotations

import json
from typing import Any, Dict

from app.config.settings import Config
from app.services.llm.gateway import LLMGateway


class GatewayFactory:
    def __init__(self):
        self._cache: Dict[str, LLMGateway] = {}

    def default_gateway(self) -> LLMGateway:
        return self.get_gateway({})

    def get_gateway(self, provider_snapshot: Dict[str, Any]) -> LLMGateway:
        runtime_config = self._build_runtime_config(provider_snapshot or {})
        cache_key = json.dumps(runtime_config, sort_keys=True, default=str)
        if cache_key not in self._cache:
            self._cache[cache_key] = LLMGateway(runtime_config=runtime_config)
        return self._cache[cache_key]

    def _build_runtime_config(self, provider_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "provider_name": (provider_snapshot.get("provider_type") or Config.LLM_PROVIDER or "openai").lower(),
            "api_key": provider_snapshot.get("api_key", Config.LLM_API_KEY),
            "base_url": provider_snapshot.get("base_url", Config.LLM_BASE_URL),
            "default_model": provider_snapshot.get("default_model", Config.LLM_EVALUATION_MODEL),
            "local_backend": provider_snapshot.get("local_backend", getattr(Config, "LOCAL_BACKEND", "")),
            "routing_config": getattr(Config, "LLM_ROUTING", {}) or {},
            "model_registry_overrides": provider_snapshot.get(
                "model_registry_overrides",
                getattr(Config, "MODEL_REGISTRY_OVERRIDES", {}) or {},
            ),
            "trace_dir": Config.TRACE_DIR,
            "trace_enabled": Config.TRACE_ENABLED,
            "trace_embed_mode": Config.TRACE_EMBED_MODE,
            "trace_embed_max_bytes": Config.TRACE_EMBED_MAX_BYTES,
            "trace_stacktrace_max_bytes": Config.TRACE_STACKTRACE_MAX_BYTES,
            "max_concurrency": Config.LLM_MAX_CONCURRENCY,
            "requests_per_minute": Config.LLM_REQUESTS_PER_MINUTE,
            "tokens_per_minute": Config.LLM_TOKENS_PER_MINUTE,
            "max_retries": Config.LLM_MAX_RETRIES,
            "backoff_base_seconds": Config.LLM_BACKOFF_BASE_SECONDS,
            "backoff_max_seconds": Config.LLM_BACKOFF_MAX_SECONDS,
            "backoff_jitter": Config.LLM_BACKOFF_JITTER,
            "timeout_seconds": Config.LLM_TIMEOUT_SECONDS,
            "timeout_seconds_local": getattr(Config, "LLM_TIMEOUT_SECONDS_LOCAL", None),
            "context_safety_margin_tokens": getattr(Config, "LLM_CONTEXT_SAFETY_MARGIN_TOKENS", 256),
            "azure_openai_endpoint": provider_snapshot.get("azure_openai_endpoint", Config.AZURE_OPENAI_ENDPOINT),
            "azure_openai_api_version": provider_snapshot.get(
                "azure_openai_api_version",
                Config.AZURE_OPENAI_API_VERSION,
            ),
            "azure_ai_inference_endpoint": provider_snapshot.get(
                "azure_ai_inference_endpoint",
                Config.AZURE_AI_INFERENCE_ENDPOINT,
            ),
            "http_referer": provider_snapshot.get("http_referer", getattr(Config, "LLM_HTTP_REFERER", None)),
            "site_name": provider_snapshot.get("site_name", getattr(Config, "LLM_SITE_NAME", None)),
            "local_model_context_override": getattr(Config, "LOCAL_MODEL_CONTEXT_OVERRIDE", None),
            "ollama_show_url": getattr(Config, "OLLAMA_SHOW_URL", None),
        }
