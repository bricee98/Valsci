import json
from pathlib import Path

from app.config.settings import Config
from app.services.provider_catalog import ProviderCatalog


def test_seed_catalog_uses_configured_model_for_ollama_provider(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(Config, "LLM_BASE_URL", "http://localhost:11434/v1", raising=False)
    monkeypatch.setattr(Config, "LLM_EVALUATION_MODEL", "llama3.1:8b", raising=False)
    monkeypatch.setattr(Config, "MODEL_REGISTRY_OVERRIDES", {}, raising=False)

    catalog = ProviderCatalog(str(tmp_path / "provider_catalog.json"))

    providers = catalog.list_providers()

    assert len(providers) == 1
    provider = providers[0]
    assert provider["provider_type"] == "ollama"
    assert provider["default_model"] == "llama3.1:8b"
    assert [model["model_name"] for model in provider["models"]] == ["llama3.1:8b"]


def test_ollama_default_provider_repairs_openai_seeded_model_list(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(Config, "LLM_EVALUATION_MODEL", "llama3.1:8b", raising=False)
    monkeypatch.setattr(Config, "MODEL_REGISTRY_OVERRIDES", {}, raising=False)
    catalog_path = tmp_path / "provider_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "provider_id": "default",
                        "label": "Default Config Provider",
                        "provider_type": "ollama",
                        "enabled": True,
                        "base_url": "http://localhost:11434/v1",
                        "default_model": "llama3.1:8b",
                        "models": [
                            {"model_name": "gpt-4o", "label": "gpt-4o"},
                            {"model_name": "gpt-5", "label": "gpt-5"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    provider = ProviderCatalog(str(catalog_path)).list_providers()[0]

    assert [model["model_name"] for model in provider["models"]] == ["llama3.1:8b"]
    saved = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert saved["providers"][0]["models"][0]["model_name"] == "llama3.1:8b"
