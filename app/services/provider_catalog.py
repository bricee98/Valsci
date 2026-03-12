"""Repo-local provider and model catalog for mixed-provider runs."""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.settings import Config
from app.services.llm.model_registry import DEFAULT_MODEL_INFO, ModelInfo


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _dedupe_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for model in models:
        name = str(model.get("model_name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized = dict(model)
        normalized.setdefault("label", name)
        normalized.setdefault("enabled", True)
        deduped.append(normalized)
    return deduped


class ProviderCatalog:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or Config.PROVIDER_CATALOG_PATH)

    def ensure_exists(self) -> None:
        if self.path.exists():
            return
        self.save(self._seed_catalog())

    def load(self) -> Dict[str, Any]:
        self.ensure_exists()
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("providers", [])
        return payload

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(payload)
        payload.setdefault("providers", [])
        for provider in payload["providers"]:
            provider.setdefault("provider_id", uuid.uuid4().hex[:10])
            provider["models"] = _dedupe_models(provider.get("models", []))
        _atomic_write_json(self.path, payload)
        return payload

    def list_providers(self) -> List[Dict[str, Any]]:
        payload = self.load()
        providers = payload.get("providers", [])
        providers.sort(key=lambda item: (str(item.get("label", "")), str(item.get("provider_id", ""))))
        return providers

    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        for provider in self.list_providers():
            if provider.get("provider_id") == provider_id:
                return provider
        return None

    @staticmethod
    def enabled_models(provider: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            dict(model)
            for model in provider.get("models", [])
            if isinstance(model, dict) and model.get("enabled", True)
        ]

    @staticmethod
    def merge_models(existing_models: List[Dict[str, Any]], incoming_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = {str(model.get("model_name", "")).strip(): dict(model) for model in existing_models if model.get("model_name")}
        for model in incoming_models:
            model_name = str(model.get("model_name", "")).strip()
            if not model_name:
                continue
            if model_name not in merged:
                merged[model_name] = dict(model)
                merged[model_name].setdefault("label", model_name)
                merged[model_name].setdefault("enabled", True)
        return _dedupe_models(list(merged.values()))

    def upsert_provider(self, provider_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.load()
        providers = payload.get("providers", [])
        provider = copy.deepcopy(provider_payload)
        provider.setdefault("provider_id", uuid.uuid4().hex[:10])
        provider.setdefault("enabled", True)
        provider.setdefault("task_defaults", {})
        provider["models"] = _dedupe_models(provider.get("models", []))
        replaced = False
        for index, existing in enumerate(providers):
            if existing.get("provider_id") == provider["provider_id"]:
                providers[index] = provider
                replaced = True
                break
        if not replaced:
            providers.append(provider)
        payload["providers"] = providers
        self.save(payload)
        return provider

    def delete_provider(self, provider_id: str) -> bool:
        payload = self.load()
        providers = payload.get("providers", [])
        updated = [provider for provider in providers if provider.get("provider_id") != provider_id]
        if len(updated) == len(providers):
            return False
        payload["providers"] = updated
        self.save(payload)
        return True

    def build_snapshot(self, provider_id: str) -> Dict[str, Any]:
        provider = self.get_provider(provider_id)
        if not provider:
            raise KeyError(f"Provider not found: {provider_id}")
        snapshot = {
            "provider_id": provider.get("provider_id"),
            "label": provider.get("label"),
            "provider_type": provider.get("provider_type"),
            "api_key": provider.get("api_key", ""),
            "base_url": provider.get("base_url"),
            "local_backend": provider.get("local_backend", ""),
            "default_model": provider.get("default_model"),
            "task_defaults": provider.get("task_defaults") or {},
            "http_referer": provider.get("http_referer"),
            "site_name": provider.get("site_name"),
            "azure_openai_endpoint": provider.get("azure_openai_endpoint"),
            "azure_openai_api_version": provider.get("azure_openai_api_version", "2024-06-01"),
            "azure_ai_inference_endpoint": provider.get("azure_ai_inference_endpoint"),
            "models": provider.get("models", []),
            "model_registry_overrides": self.model_registry_overrides_for_provider(provider),
        }
        return snapshot

    def model_registry_overrides_for_provider(self, provider: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        overrides: Dict[str, Dict[str, Any]] = {}
        for model in provider.get("models", []):
            model_name = model.get("model_name")
            if not model_name:
                continue
            overrides[model_name] = {
                "context_window_tokens": int(model.get("context_window_tokens", 8192)),
                "max_output_tokens_default": int(model.get("max_output_tokens_default", 1024)),
                "supports_temperature": bool(model.get("supports_temperature", True)),
                "supports_json_schema_or_json_mode": bool(model.get("supports_json_mode", True)),
                "input_cost_per_million": float(model.get("input_cost_per_million", 0.0)),
                "output_cost_per_million": float(model.get("output_cost_per_million", 0.0)),
            }
        return overrides

    def _seed_catalog(self) -> Dict[str, Any]:
        model_entries: List[Dict[str, Any]] = []
        for model_name, model_info in DEFAULT_MODEL_INFO.items():
            model_entries.append(self._model_entry(model_name, model_info))
        overrides = getattr(Config, "MODEL_REGISTRY_OVERRIDES", {}) or {}
        for model_name, values in overrides.items():
            model_entries.append(
                {
                    "model_name": model_name,
                    "label": model_name,
                    "enabled": True,
                    "context_window_tokens": int(values.get("context_window_tokens", 8192)),
                    "max_output_tokens_default": int(values.get("max_output_tokens_default", 1024)),
                    "supports_temperature": bool(values.get("supports_temperature", True)),
                    "supports_json_mode": bool(
                        values.get(
                            "supports_json_schema_or_json_mode",
                            values.get("supports_json_mode", True),
                        )
                    ),
                    "input_cost_per_million": float(values.get("input_cost_per_million", 0.0)),
                    "output_cost_per_million": float(values.get("output_cost_per_million", 0.0)),
                }
            )

        provider = {
            "provider_id": "default",
            "label": "Default Config Provider",
            "provider_type": Config.LLM_PROVIDER,
            "enabled": True,
            "api_key": Config.LLM_API_KEY,
            "base_url": Config.LLM_BASE_URL,
            "local_backend": getattr(Config, "LOCAL_BACKEND", ""),
            "default_model": Config.LLM_EVALUATION_MODEL,
            "task_defaults": {},
            "http_referer": getattr(Config, "LLM_HTTP_REFERER", None),
            "site_name": getattr(Config, "LLM_SITE_NAME", None),
            "azure_openai_endpoint": Config.AZURE_OPENAI_ENDPOINT,
            "azure_openai_api_version": Config.AZURE_OPENAI_API_VERSION,
            "azure_ai_inference_endpoint": Config.AZURE_AI_INFERENCE_ENDPOINT,
            "models": _dedupe_models(model_entries),
        }
        return {"providers": [provider]}

    @staticmethod
    def _model_entry(model_name: str, model_info: ModelInfo) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "label": model_name,
            "enabled": True,
            "context_window_tokens": model_info.context_window_tokens,
            "max_output_tokens_default": model_info.max_output_tokens_default,
            "supports_temperature": model_info.supports_temperature,
            "supports_json_mode": model_info.supports_json_mode,
            "input_cost_per_million": model_info.input_cost_per_million,
            "output_cost_per_million": model_info.output_cost_per_million,
        }
