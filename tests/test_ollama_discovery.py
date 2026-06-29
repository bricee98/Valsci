import sys
import types
from pathlib import Path

import pytest
import requests

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault(
    "openai",
    types.SimpleNamespace(
        OpenAI=object,
        AsyncOpenAI=object,
        AsyncAzureOpenAI=object,
    ),
)

from app import create_app
from app.api import routes as routes_module
from app.config.settings import Config
from app.services import ollama_discovery
from app.services.ollama_discovery import (
    DEFAULT_OLLAMA_BASE_URL,
    discover_ollama_models,
    normalize_ollama_base_url,
    parse_ollama_list_output,
)
from app.services.provider_catalog import ProviderCatalog


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.payload


def test_parse_ollama_list_output_extracts_models():
    payload = """NAME            ID              SIZE      MODIFIED
llama3.2:latest  abcdef123456    2.0 GB    2 hours ago
phi4-mini        fedcba654321    8.8 GB    3 days ago
"""

    models = parse_ollama_list_output(payload)

    assert [model["model_name"] for model in models] == ["llama3.2:latest", "phi4-mini"]
    assert models[0]["discovery_metadata"]["size"] == "2.0 GB"
    assert models[0]["discovery_metadata"]["tag"] == "latest"


def test_normalize_ollama_base_url_trims_runtime_suffixes():
    assert normalize_ollama_base_url("localhost:11434/api") == DEFAULT_OLLAMA_BASE_URL
    assert normalize_ollama_base_url("http://localhost:11434/v1/") == DEFAULT_OLLAMA_BASE_URL
    assert normalize_ollama_base_url("http://localhost:11434/custom") == "http://localhost:11434/custom"


def test_discover_ollama_models_uses_http_tags_and_show(monkeypatch):
    calls = []

    def fake_request(method, url, headers=None, json=None, timeout=None):
      calls.append((method, url, json))
      if url.endswith("/api/tags"):
          return FakeResponse(
              {
                  "models": [
                      {
                          "name": "llama3.2:latest",
                          "size": 2147483648,
                          "modified_at": "2026-03-13T12:00:00Z",
                          "digest": "sha256:abc",
                      }
                  ]
              }
          )
      if url.endswith("/api/show"):
          return FakeResponse(
              {
                  "details": {
                      "context_length": 32768,
                      "parameter_size": "8B",
                      "family": "llama",
                      "quantization_level": "Q4_K_M",
                  },
                  "capabilities": ["completion", "json"],
              }
          )
      raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(requests, "request", fake_request)

    models = discover_ollama_models(base_url="localhost:11434/api")

    assert len(models) == 1
    assert models[0]["model_name"] == "llama3.2:latest"
    assert models[0]["context_window_tokens"] == 32768
    assert models[0]["discovery_metadata"]["family"] == "llama"
    assert calls[0][0] == "GET"
    assert calls[0][1] == "http://localhost:11434/api/tags"
    assert calls[1][0] == "POST"
    assert calls[1][2] == {"model": "llama3.2:latest"}


def test_discover_ollama_models_raises_when_host_is_unreachable(monkeypatch):
    def fake_request(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="Could not reach an Ollama host"):
        discover_ollama_models(base_url="http://localhost:11434")


def test_unreachable_localhost_inside_docker_suggests_host_docker_internal(monkeypatch):
    monkeypatch.setenv("VALSCI_IN_DOCKER", "1")

    def fake_request(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="host.docker.internal"):
        discover_ollama_models(base_url="http://localhost:11434")
    with pytest.raises(RuntimeError, match="host.docker.internal"):
        discover_ollama_models(base_url="http://127.0.0.1:11434")


def test_unreachable_host_docker_internal_outside_docker_suggests_localhost(monkeypatch):
    monkeypatch.delenv("VALSCI_IN_DOCKER", raising=False)
    monkeypatch.setattr(ollama_discovery, "running_in_docker", lambda: False)

    def fake_request(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="use\\s+http://localhost:11434 instead"):
        discover_ollama_models(base_url="http://host.docker.internal:11434")


def test_unreachable_remote_host_gets_no_docker_hint(monkeypatch):
    monkeypatch.setenv("VALSCI_IN_DOCKER", "1")

    def fake_request(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "request", fake_request)

    with pytest.raises(RuntimeError) as excinfo:
        discover_ollama_models(base_url="http://192.168.1.50:11434")
    assert "host.docker.internal" not in str(excinfo.value)


def test_discovery_timeout_includes_docker_hint(monkeypatch):
    monkeypatch.setenv("VALSCI_IN_DOCKER", "1")

    def fake_request(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="timed out.*host.docker.internal"):
        discover_ollama_models(base_url="http://localhost:11434")


def test_merge_models_preserves_existing_manual_metadata():
    existing = [
        {
            "model_name": "llama3.2:latest",
            "label": "Curated Llama",
            "enabled": False,
            "context_window_tokens": 32768,
            "max_output_tokens_default": 2048,
            "input_cost_per_million": 1.2,
            "output_cost_per_million": 3.4,
        }
    ]
    incoming = [
        {
            "model_name": "llama3.2:latest",
            "label": "Discovered Llama",
            "enabled": True,
            "context_window_tokens": 8192,
            "max_output_tokens_default": 1024,
        },
        {
            "model_name": "phi4-mini",
            "label": "phi4-mini",
            "enabled": True,
            "context_window_tokens": 8192,
            "max_output_tokens_default": 1024,
        },
    ]

    merged = ProviderCatalog.merge_models(existing, incoming)

    assert len(merged) == 2
    preserved = next(model for model in merged if model["model_name"] == "llama3.2:latest")
    assert preserved["label"] == "Curated Llama"
    assert preserved["enabled"] is False
    assert preserved["context_window_tokens"] == 32768


def test_discovery_route_accepts_explicit_base_url(monkeypatch, tmp_path: Path):
    catalog_path = tmp_path / "provider_catalog.json"
    monkeypatch.setattr(Config, "PROVIDER_CATALOG_PATH", str(catalog_path), raising=False)
    monkeypatch.setattr(TestConfig, "PROVIDER_CATALOG_PATH", str(catalog_path), raising=False)

    app = create_app(TestConfig)
    client = app.test_client()

    monkeypatch.setattr(
        routes_module,
        "discover_ollama_models",
        lambda base_url=None, api_key=None: [{"model_name": "phi4-mini", "context_window_tokens": 8192}],
    )

    response = client.post(
        "/api/v1/providers/ollama/discover",
        json={"base_url": "http://localhost:11434"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["base_url"] == "http://localhost:11434"


def test_providers_page_renders_http_discovery_editor(monkeypatch, tmp_path):
    catalog_path = tmp_path / "provider_catalog.json"
    monkeypatch.setattr(Config, "PROVIDER_CATALOG_PATH", str(catalog_path), raising=False)
    monkeypatch.setattr(TestConfig, "PROVIDER_CATALOG_PATH", str(catalog_path), raising=False)

    app = create_app(TestConfig)
    client = app.test_client()

    response = client.get("/providers")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Connection" in page
    assert "Discover Models" in page
    assert "Probe Alternate URL" in page
