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
from app.services.llm.types import empty_usage
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


def seed_stage_checkpoint_run(
    store: ClaimStore,
    saved_jobs_dir: Path,
    *,
    arena_id: str,
    claim_text: str,
    provider_snapshot: dict,
    stop_after: str = "query_generation",
):
    claim_record, _ = store.get_or_create_claim(claim_text, batch_tags=[arena_id])
    run_record = store.create_run(
        claim_record=claim_record,
        batch_tags=[arena_id],
        arena_id=arena_id,
        execution_mode="full_pipeline",
        stop_after=stop_after,
        provider_snapshot=provider_snapshot,
        cost_confirmation={"accepted": True},
        transport_batch_id=arena_id,
        review_type="regular",
        status="processed",
        source="arena",
        completed_stage=stop_after,
        is_stage_checkpoint=(stop_after != "final_report"),
    )
    claim_data = {
        "text": claim_text,
        "status": "processed",
        "batch_id": arena_id,
        "claim_id": run_record["run_id"],
        "run_id": run_record["run_id"],
        "claim_key": claim_record["claim_key"],
        "arena_id": arena_id,
        "review_type": "regular",
        "execution_mode": "full_pipeline",
        "stop_after": stop_after,
        "completed_stage": stop_after,
        "is_stage_checkpoint": stop_after != "final_report",
        "provider_snapshot": provider_snapshot,
        "model_overrides": {},
        "search_config": {"num_queries": 5, "results_per_query": 5},
        "bibliometric_config": {"use_bibliometrics": True},
        "usage": empty_usage(),
        "usage_by_stage": {},
        "semantic_scholar_queries": ["query one", "query two"],
    }
    if stop_after in {"paper_analysis", "venue_scoring", "final_report"}:
        claim_data["raw_papers"] = [{"corpusId": 1, "title": "Paper A"}]
        claim_data["processed_papers"] = [
            {
                "paper": {"corpusId": 1, "title": "Paper A"},
                "relevance": 0.8,
                "excerpts": ["Excerpt"],
                "explanations": ["Explanation"],
                "score": 0.6 if stop_after in {"venue_scoring", "final_report"} else -1,
                "score_status": "completed" if stop_after in {"venue_scoring", "final_report"} else "pending",
            }
        ]
        claim_data["non_relevant_papers"] = []
        claim_data["inaccessible_papers"] = []
        claim_data["failed_papers"] = []
    claim_path = saved_jobs_dir / arena_id / f"{run_record['run_id']}.txt"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(json.dumps(claim_data, indent=2), encoding="utf-8")
    store.ingest_transport_artifact(arena_id, run_record["run_id"])
    return store.get_run(run_record["run_id"])


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


def test_preflight_stop_after_skips_later_stages(monkeypatch, tmp_path):
    client, _, _, _, _ = create_test_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/claims/preflight",
        json={
            "claims": ["Creatine improves short-term memory."],
            "stop_after": "paper_analysis",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    stages = payload["candidates"][0]["runs"][0]["estimate"]["expected"]["stages"]
    assert payload["stop_after"] == "paper_analysis"
    assert stages["query_generation"]["skipped"] is False
    assert stages["venue_scoring"]["skipped"] is True
    assert stages["final_report"]["skipped"] is True


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


def test_create_arena_persists_stage_history_for_stop_after(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/arenas",
        json={
            "title": "Query Stage Arena",
            "claims": ["Creatine improves short-term memory."],
            "candidates": [{"provider_id": "default"}],
            "execution_mode": "reuse_retrieval",
            "stop_after": "query_generation",
            "cost_confirmation": {"accepted": True},
        },
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["execution_mode"] == "full_pipeline"
    assert payload["stop_after"] == "query_generation"

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    arena = store.get_arena(payload["arena_id"])
    assert arena["current_stage"] == "query_generation"
    assert arena["stage_history"][-1]["stage"] == "query_generation"
    run = store.get_run(payload["created_runs"][0]["run_id"])
    assert run["stop_after"] == "query_generation"


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


def test_batch_state_counts_stage_checkpoint_as_completed(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    arena = store.create_arena(title="Checkpoint Arena", batch_tags=["checkpoint-batch"], current_stage="query_generation")
    run = seed_stage_checkpoint_run(
        store,
        saved_jobs_dir,
        arena_id=arena["arena_id"],
        claim_text="Creatine improves memory.",
        provider_snapshot=ProviderCatalog(str(provider_catalog_path)).build_snapshot("default"),
        stop_after="query_generation",
    )
    arena["claim_keys"] = [run["claim_key"]]
    store.append_arena_stage_history(
        arena,
        stage="query_generation",
        run_ids=[run["run_id"]],
        source="initial_submit",
    )

    batch_state = store.build_batch_state(arena["arena_id"])

    assert batch_state["status"] == "completed"
    assert batch_state["processed_claims"] == 1
    assert batch_state["claims"][0]["checkpoint_complete"] is True


def test_continue_arena_endpoints_create_seeded_next_stage_runs(monkeypatch, tmp_path):
    client, saved_jobs_dir, queued_jobs_dir, state_dir, provider_catalog_path = create_test_client(monkeypatch, tmp_path)
    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    arena = store.create_arena(
        title="Checkpoint Arena",
        batch_tags=["checkpoint-batch"],
        execution_mode="full_pipeline",
        current_stage="query_generation",
        candidates=[
            {"candidate_id": "candidate-0", "provider_id": "default", "label": "Baseline"},
            {"candidate_id": "candidate-1", "provider_id": "default", "label": "Alt"},
        ],
    )
    default_snapshot = ProviderCatalog(str(provider_catalog_path)).build_snapshot("default")
    claim_one_run_a = seed_stage_checkpoint_run(
        store,
        saved_jobs_dir,
        arena_id=arena["arena_id"],
        claim_text="Creatine improves memory.",
        provider_snapshot={**default_snapshot, "label": "Baseline"},
        stop_after="query_generation",
    )
    claim_one_run_b = seed_stage_checkpoint_run(
        store,
        saved_jobs_dir,
        arena_id=arena["arena_id"],
        claim_text="Creatine improves memory.",
        provider_snapshot={**default_snapshot, "label": "Alt"},
        stop_after="query_generation",
    )
    claim_two_run = seed_stage_checkpoint_run(
        store,
        saved_jobs_dir,
        arena_id=arena["arena_id"],
        claim_text="Vitamin D reduces falls.",
        provider_snapshot={**default_snapshot, "label": "Baseline"},
        stop_after="query_generation",
    )
    arena["claim_keys"] = [claim_one_run_a["claim_key"], claim_two_run["claim_key"]]
    store.append_arena_stage_history(
        arena,
        stage="query_generation",
        run_ids=[claim_one_run_a["run_id"], claim_one_run_b["run_id"], claim_two_run["run_id"]],
        source="initial_submit",
    )

    preflight_response = client.post(
        f"/api/v1/arenas/{arena['arena_id']}/continue/preflight",
        json={
            "decisions": [
                {"claim_key": claim_one_run_a["claim_key"], "selected_run_id": claim_one_run_b["run_id"]},
                {"claim_key": claim_two_run["claim_key"], "skip_claim": True},
            ]
        },
    )

    assert preflight_response.status_code == 200
    preflight_payload = preflight_response.get_json()
    assert preflight_payload["next_stage"] == "paper_analysis"
    assert preflight_payload["totals"]["run_count"] == 1

    continue_response = client.post(
        f"/api/v1/arenas/{arena['arena_id']}/continue",
        json={
            "decisions": [
                {"claim_key": claim_one_run_a["claim_key"], "selected_run_id": claim_one_run_b["run_id"]},
                {"claim_key": claim_two_run["claim_key"], "skip_claim": True},
            ],
            "cost_confirmation": {"accepted": True},
        },
    )

    assert continue_response.status_code == 202
    continue_payload = continue_response.get_json()
    assert len(continue_payload["created_runs"]) == 1
    created_run = continue_payload["created_runs"][0]
    assert created_run["stop_after"] == "paper_analysis"
    assert created_run["seed_from_run_id"] == claim_one_run_b["run_id"]

    queued_path = queued_jobs_dir / arena["arena_id"] / f"{created_run['run_id']}.txt"
    queued_payload = json.loads(queued_path.read_text(encoding="utf-8"))
    assert queued_payload["status"] == "ready_for_search"
    assert queued_payload["seed_from_run_id"] == claim_one_run_b["run_id"]
    assert queued_payload["stop_after"] == "paper_analysis"

    updated_arena = store.get_arena(arena["arena_id"])
    assert updated_arena["current_stage"] == "paper_analysis"
    assert updated_arena["stage_history"][-1]["stage"] == "paper_analysis"

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
