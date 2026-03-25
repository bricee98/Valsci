"""Helpers for discovering Ollama models over HTTP."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import requests


TABLE_SPLIT_RE = re.compile(r"\s{2,}")
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
_PATH_SUFFIXES_TO_TRIM = ("/api", "/api/", "/v1", "/v1/")


def parse_ollama_list_output(output: str) -> List[Dict[str, Any]]:
    lines = [line.rstrip() for line in str(output or "").splitlines() if line.strip()]
    if not lines:
        return []
    if lines[0].strip().lower().startswith("name"):
        lines = lines[1:]

    discovered: List[Dict[str, Any]] = []
    for line in lines:
        parts = TABLE_SPLIT_RE.split(line.strip())
        if not parts:
            continue
        model_name = str(parts[0]).strip()
        if not model_name:
            continue
        tag = None
        if ":" in model_name:
            _, tag = model_name.rsplit(":", 1)
        discovered.append(
            {
                "model_name": model_name,
                "label": model_name,
                "enabled": True,
                "context_window_tokens": 8192,
                "max_output_tokens_default": 1024,
                "supports_temperature": True,
                "supports_json_mode": True,
                "input_cost_per_million": 0.0,
                "output_cost_per_million": 0.0,
                "discovery_metadata": {
                    "model_id": parts[1] if len(parts) > 1 else None,
                    "size": parts[2] if len(parts) > 2 else None,
                    "modified": parts[3] if len(parts) > 3 else None,
                    "tag": tag,
                },
            }
        )
    return discovered


def normalize_ollama_base_url(base_url: Optional[str]) -> str:
    raw = str(base_url or DEFAULT_OLLAMA_BASE_URL).strip()
    if not raw:
        raw = DEFAULT_OLLAMA_BASE_URL
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    lowered = path.lower()
    for suffix in _PATH_SUFFIXES_TO_TRIM:
        if lowered.endswith(suffix.rstrip("/")):
            path = path[: -len(suffix.rstrip("/"))]
            break
    normalized = parsed._replace(path=path or "", params="", query="", fragment="")
    return urlunparse(normalized).rstrip("/")


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_json(
    method: str,
    url: str,
    *,
    api_key: Optional[str],
    timeout_seconds: int,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = requests.request(
        method=method,
        url=url,
        headers=_headers(api_key),
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Ollama endpoint returned an unexpected payload from {url}.")
    return data


def probe_ollama_host(
    *,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout_seconds: int = 10,
) -> Dict[str, Any]:
    normalized_base_url = normalize_ollama_base_url(base_url)
    try:
        tags_payload = _request_json(
            "GET",
            f"{normalized_base_url}/api/tags",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise RuntimeError("Ollama model discovery timed out.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach an Ollama host at {normalized_base_url}: {exc}") from exc

    models = tags_payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError(f"Host at {normalized_base_url} did not respond like Ollama.")

    return {
        "base_url": normalized_base_url,
        "models": models,
    }


def _extract_context_window(show_payload: Dict[str, Any]) -> Optional[int]:
    details = show_payload.get("details")
    if isinstance(details, dict):
        for key in ["context_length", "num_ctx"]:
            value = details.get(key)
            try:
                if value is not None:
                    return int(value)
            except Exception:
                pass

    model_info = show_payload.get("model_info")
    if isinstance(model_info, dict):
        candidate_keys = [
            "llama.context_length",
            "qwen2.context_length",
            "mistral.context_length",
            "gemma.context_length",
            "num_ctx",
        ]
        for key in candidate_keys:
            value = model_info.get(key)
            try:
                if value is not None:
                    return int(value)
            except Exception:
                pass

    parameters = show_payload.get("parameters")
    if isinstance(parameters, str):
        match = re.search(r"num_ctx\s+(\d+)", parameters)
        if match:
            return int(match.group(1))
    return None


def _extract_capabilities(show_payload: Dict[str, Any]) -> List[str]:
    capabilities = show_payload.get("capabilities")
    if isinstance(capabilities, list):
        return [str(value).strip() for value in capabilities if str(value).strip()]
    return []


def discover_ollama_models(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: int = 10,
) -> List[Dict[str, Any]]:
    probe = probe_ollama_host(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    normalized_base_url = probe["base_url"]
    discovered: List[Dict[str, Any]] = []

    for model in probe["models"]:
        if not isinstance(model, dict):
            continue
        model_name = str(model.get("name") or "").strip()
        if not model_name:
            continue

        show_payload: Dict[str, Any] = {}
        try:
            show_payload = _request_json(
                "POST",
                f"{normalized_base_url}/api/show",
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                payload={"model": model_name},
            )
        except requests.RequestException:
            show_payload = {}

        tag = None
        if ":" in model_name:
            _, tag = model_name.rsplit(":", 1)

        context_window = _extract_context_window(show_payload) or 8192
        capabilities = _extract_capabilities(show_payload)
        supports_json_mode = any("json" in capability.lower() for capability in capabilities)
        modified_at = model.get("modified_at") or model.get("modified")

        discovered.append(
            {
                "model_name": model_name,
                "label": model_name,
                "enabled": True,
                "context_window_tokens": context_window,
                "max_output_tokens_default": min(max(context_window // 4, 512), 8192),
                "supports_temperature": True,
                "supports_json_mode": supports_json_mode or True,
                "input_cost_per_million": 0.0,
                "output_cost_per_million": 0.0,
                "discovery_metadata": {
                    "tag": tag,
                    "size": model.get("size"),
                    "modified": modified_at,
                    "digest": model.get("digest"),
                    "parameter_size": (show_payload.get("details") or {}).get("parameter_size"),
                    "family": (show_payload.get("details") or {}).get("family"),
                    "quantization_level": (show_payload.get("details") or {}).get("quantization_level"),
                    "capabilities": capabilities,
                    "base_url": normalized_base_url,
                },
            }
        )

    return discovered
