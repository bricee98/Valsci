import subprocess
import sys
import types

import pytest

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
from app.config.settings import Config
from app.services.ollama_discovery import discover_ollama_models, parse_ollama_list_output
from app.services.provider_catalog import ProviderCatalog


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


def test_parse_ollama_list_output_extracts_models():
    payload = """NAME            ID              SIZE      MODIFIED
llama3.2:latest  abcdef123456    2.0 GB    2 hours ago
phi4-mini        fedcba654321    8.8 GB    3 days ago
"""

    models = parse_ollama_list_output(payload)

    assert [model["model_name"] for model in models] == ["llama3.2:latest", "phi4-mini"]
    assert models[0]["discovery_metadata"]["size"] == "2.0 GB"
    assert models[0]["discovery_metadata"]["tag"] == "latest"


def test_discover_ollama_models_raises_when_cli_missing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", missing)

    with pytest.raises(RuntimeError, match="not found"):
        discover_ollama_models()


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


def test_providers_page_renders_responsive_editor(monkeypatch, tmp_path):
    catalog_path = tmp_path / "provider_catalog.json"
    monkeypatch.setattr(Config, "PROVIDER_CATALOG_PATH", str(catalog_path), raising=False)
    monkeypatch.setattr(TestConfig, "PROVIDER_CATALOG_PATH", str(catalog_path), raising=False)

    app = create_app(TestConfig)
    client = app.test_client()

    response = client.get("/providers")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Find provider" in page
    assert "Connection" in page
    assert "Discover Ollama Models" in page
