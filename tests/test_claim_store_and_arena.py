import json
import sys
import types
from pathlib import Path

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app import create_app
from app.api import routes as routes_module
from app.config.settings import Config
from app.services.claim_store import ClaimStore, claim_key_for_text
from app.services.provider_catalog import ProviderCatalog


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


def create_test_client(monkeypatch, tmp_path: Path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    state_dir = tmp_path / "state"
    provider_catalog_path = state_dir / "provider_catalog.json"

    monkeypatch.setattr(routes_module, "SAVED_JOBS_DIR", str(saved_jobs_dir))
    monkeypatch.setattr(routes_module, "QUEUED_JOBS_DIR", str(queued_jobs_dir))
    monkeypatch.setattr(Config, "SAVED_JOBS_DIR", str(saved_jobs_dir), raising=False)
    monkeypatch.setattr(Config, "QUEUED_JOBS_DIR", str(queued_jobs_dir), raising=False)
    monkeypatch.setattr(Config, "TRACE_DIR", str(saved_jobs_dir), raising=False)
    monkeypatch.setattr(Config, "STATE_DIR", str(state_dir), raising=False)
    monkeypatch.setattr(Config, "PROVIDER_CATALOG_PATH", str(provider_catalog_path), raising=False)

    app = create_app(TestConfig)
    app.config["SAVED_JOBS_DIR"] = str(saved_jobs_dir)
    app.config["QUEUED_JOBS_DIR"] = str(queued_jobs_dir)
    app.config["TRACE_DIR"] = str(saved_jobs_dir)
    app.config["STATE_DIR"] = str(state_dir)
    app.config["PROVIDER_CATALOG_PATH"] = str(provider_catalog_path)
    return app.test_client(), saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path


def test_claim_key_normalization_collapses_case_and_whitespace():
    assert claim_key_for_text("  Creatine improves memory  ") == claim_key_for_text("creatine   improves\nmemory")
    assert claim_key_for_text("Creatine improves memory.") != claim_key_for_text("Creatine improves memory")


def test_preflight_reports_duplicates_and_existing_claim(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    claim, _ = store.get_or_create_claim("Creatine improves memory in adults.", batch_tags=["batch-a"])
    store.create_run(
        claim_record=claim,
        batch_tags=["batch-a"],
        provider_snapshot=ProviderCatalog(str(provider_catalog_path)).build_snapshot("default"),
        cost_confirmation={"accepted": True},
    )

    response = client.post(
        "/api/v1/claims/preflight",
        json={
            "claims": [
                "Creatine improves memory in adults.",
                " creatine   improves memory in adults. ",
                "Vitamin D reduces falls.",
            ]
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["duplicates"]) == 1
    assert payload["claims"][0]["existing_claim"] is True
    assert payload["claims"][1]["existing_claim"] is True
    assert payload["totals"]["claim_count"] == 3
    assert payload["totals"]["unique_claim_count"] == 2
    assert payload["totals"]["run_count"] == 2


def test_preflight_view_existing_reuses_latest_run_and_charges_only_new_claims(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    claim, _ = store.get_or_create_claim("Creatine improves memory in adults.", batch_tags=["batch-a"])
    store.create_run(
        claim_record=claim,
        batch_tags=["batch-a"],
        provider_snapshot=ProviderCatalog(str(provider_catalog_path)).build_snapshot("default"),
        cost_confirmation={"accepted": True},
    )

    response = client.post(
        "/api/v1/claims/preflight",
        json={
            "claims": [
                "Creatine improves memory in adults.",
                "Vitamin D reduces falls.",
            ],
            "duplicate_strategy": "view",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["totals"]["unique_claim_count"] == 2
    assert payload["totals"]["reused_existing_count"] == 1
    assert payload["totals"]["run_count"] == 1
    run_actions = {run["text"]: run["action"] for run in payload["candidates"][0]["runs"]}
    assert run_actions["Creatine improves memory in adults."] == "view_existing"
    assert run_actions["Vitamin D reduces falls."] == "create_run"


def test_reuse_retrieval_preflight_only_charges_query_generation_once(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    catalog = ProviderCatalog(str(provider_catalog_path))
    catalog.upsert_provider(
        {
            "provider_id": "openrouter-alt",
            "label": "OpenRouter Alt",
            "provider_type": "openrouter",
            "api_key": "router-key",
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "gpt-4o-mini",
            "models": [
                {
                    "model_name": "gpt-4o-mini",
                    "context_window_tokens": 128000,
                    "max_output_tokens_default": 4096,
                    "supports_temperature": True,
                    "supports_json_mode": True,
                    "input_cost_per_million": 0.15,
                    "output_cost_per_million": 0.60,
                }
            ],
        }
    )

    response = client.post(
        "/api/v1/claims/preflight",
        json={
            "claims": ["Creatine improves short-term memory."],
            "candidates": [{"provider_id": "default"}, {"provider_id": "openrouter-alt"}],
            "execution_mode": "reuse_retrieval",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    baseline_run = payload["candidates"][0]["runs"][0]
    reused_run = payload["candidates"][1]["runs"][0]
    assert baseline_run["estimate"]["expected"]["stages"]["query_generation"]["cost_usd"] > 0
    assert reused_run["estimate"]["expected"]["stages"]["query_generation"]["cost_usd"] == 0
    assert reused_run["estimate"]["expected"]["stages"]["query_generation"]["skipped"] is True


def test_create_runs_persists_state_and_queue_file(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/runs",
        json={
            "claims": ["Magnesium improves sleep quality."],
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["created_runs"]
    run_id = payload["created_runs"][0]["run_id"]
    run_file = queued_jobs_dir / payload["batch_id"] / f"{run_id}.txt"
    assert run_file.exists()

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    run_record = store.get_run(run_id)
    assert run_record is not None
    assert run_record["text"] == "Magnesium improves sleep quality."
    assert run_record["provider_snapshot"]["provider_id"] == "default"


def test_create_runs_deduplicates_duplicate_input_claims(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/runs",
        json={
            "claims": [
                "Magnesium improves sleep quality.",
                " magnesium improves   sleep quality. ",
            ],
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert len(payload["created_runs"]) == 1
    assert payload["preflight"]["totals"]["unique_claim_count"] == 1
    assert payload["preflight"]["totals"]["run_count"] == 1


def test_create_runs_rejects_unknown_remote_model_pricing(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    catalog = ProviderCatalog(str(provider_catalog_path))
    catalog.upsert_provider(
        {
            "provider_id": "openrouter-unknown",
            "label": "OpenRouter Unknown",
            "provider_type": "openrouter",
            "api_key": "router-key",
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "mystery-model",
            "models": [],
        }
    )

    response = client.post(
        "/api/v1/runs",
        json={
            "claims": ["Magnesium improves sleep quality."],
            "candidates": [{"provider_id": "openrouter-unknown"}],
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "Missing pricing metadata" in payload["error"]


def test_create_arena_supports_multiple_provider_snapshots(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    catalog = ProviderCatalog(str(provider_catalog_path))
    catalog.upsert_provider(
        {
            "provider_id": "openrouter-alt",
            "label": "OpenRouter Alt",
            "provider_type": "openrouter",
            "api_key": "router-key",
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "gpt-4o-mini",
            "models": [
                {
                    "model_name": "gpt-4o-mini",
                    "context_window_tokens": 128000,
                    "max_output_tokens_default": 4096,
                    "supports_temperature": True,
                    "supports_json_mode": True,
                    "input_cost_per_million": 0.15,
                    "output_cost_per_million": 0.60,
                }
            ],
        }
    )

    response = client.post(
        "/api/v1/arenas",
        json={
            "title": "Memory Arena",
            "claims": ["Creatine improves short-term memory."],
            "candidates": [
                {"provider_id": "default"},
                {"provider_id": "openrouter-alt"},
            ],
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["arena_id"]
    assert len(payload["created_runs"]) == 2

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    arena = store.get_arena(payload["arena_id"])
    assert arena is not None
    run_ids = [run["run_id"] for run in payload["created_runs"]]
    providers = {store.get_run(run_id)["provider_snapshot"]["provider_id"] for run_id in run_ids}
    assert providers == {"default", "openrouter-alt"}


def test_create_arena_can_compare_same_provider_with_different_candidate_labels(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/arenas",
        json={
            "title": "Same Provider Arena",
            "claims": ["Creatine improves short-term memory."],
            "candidates": [
                {"provider_id": "default", "label": "Default Baseline"},
                {
                    "provider_id": "default",
                    "label": "Default Final Only",
                    "model_overrides": {"final_report": "gpt-5"},
                },
            ],
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert len(payload["created_runs"]) == 2

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    labels = {store.get_run(run["run_id"])["provider_snapshot"]["label"] for run in payload["created_runs"]}
    assert labels == {"Default Baseline", "Default Final Only"}


def test_reuse_retrieval_only_queues_baseline_run_initially(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    catalog = ProviderCatalog(str(provider_catalog_path))
    catalog.upsert_provider(
        {
            "provider_id": "local-alt",
            "label": "Local Alt",
            "provider_type": "local",
            "base_url": "http://localhost:11434/v1",
            "local_backend": "ollama",
            "default_model": "llama3.2",
            "models": [
                {
                    "model_name": "llama3.2",
                    "context_window_tokens": 8192,
                    "max_output_tokens_default": 1024,
                    "supports_temperature": True,
                    "supports_json_mode": False,
                    "input_cost_per_million": 0.0,
                    "output_cost_per_million": 0.0,
                }
            ],
        }
    )

    response = client.post(
        "/api/v1/arenas",
        json={
            "claims": ["Blue light blocking glasses improve sleep latency."],
            "candidates": [{"provider_id": "default"}, {"provider_id": "local-alt"}],
            "execution_mode": "reuse_retrieval",
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    runs = [store.get_run(run["run_id"]) for run in payload["created_runs"]]
    statuses = sorted(run["status"] for run in runs)
    assert statuses == ["queued", "waiting_for_baseline"]
    queued_files = list(queued_jobs_dir.rglob("*.txt"))
    queued_claim_files = [path for path in queued_files if path.name != "claims.txt"]
    assert len(queued_claim_files) == 1


def test_migration_route_imports_legacy_saved_claim(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    batch_dir = saved_jobs_dir / "legacy-batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "legacy-claim.txt").write_text(
        json.dumps(
            {
                "text": "Legacy claim text",
                "status": "processed",
                "batch_id": "legacy-batch",
                "claim_id": "legacy-claim",
                "search_config": {"num_queries": 4, "results_per_query": 5},
                "bibliometric_config": {"use_bibliometrics": True},
                "report": {"claimRating": 4, "explanation": "Legacy report"},
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cost_usd": 0.01, "is_estimated": False},
                "usage_by_stage": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    status_response = client.get("/api/v1/migration/status")
    assert status_response.status_code == 200
    assert "legacy-batch" in status_response.get_json()["pending_batches"]

    response = client.post("/api/v1/migration/run", json={"apply": True})
    assert response.status_code == 200

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    run = store.find_run_by_legacy("legacy-batch", "legacy-claim")
    assert run is not None
    assert run["text"] == "Legacy claim text"
