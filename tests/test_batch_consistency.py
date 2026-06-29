import json
import sys
import types
import zipfile
from io import BytesIO
from pathlib import Path

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app import create_app
from app.api import routes as routes_module
from app.config.settings import Config
from app.services.claim_store import ClaimStore
from app.services.processor_heartbeat import write_heartbeat


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


def write_claim(
    root_dir: Path,
    batch_id: str,
    claim_id: str,
    *,
    status: str,
    text: str,
    report_available: bool,
    review_type: str = "regular",
    provider_snapshot: dict | None = None,
    model_overrides: dict | None = None,
):
    batch_dir = root_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    claim_data = {
        "text": text,
        "status": status,
        "batch_id": batch_id,
        "claim_id": claim_id,
        "review_type": review_type,
        "search_config": {"num_queries": 4, "results_per_query": 5},
        "semantic_scholar_queries": ["query one"] if status != "queued" else [],
        "raw_papers": [{"title": "Paper A", "corpusId": 1}] if status in {"processed", "ready_for_analysis"} else [],
        "processed_papers": [],
        "non_relevant_papers": [],
        "inaccessible_papers": [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "is_estimated": False,
        },
        "usage_by_stage": {},
    }
    if provider_snapshot is not None:
        claim_data["provider_snapshot"] = provider_snapshot
    if model_overrides is not None:
        claim_data["model_overrides"] = model_overrides
    if report_available:
        claim_data["report"] = {
            "claimRating": 4,
            "explanation": "Evidence leans supportive.",
            "relevantPapers": [{"title": "Paper A"}],
            "nonRelevantPapers": [],
            "usage_summary": {
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
                "cost_usd": 0.01,
                "is_estimated": False,
            },
        }

    (batch_dir / f"{claim_id}.txt").write_text(json.dumps(claim_data, indent=2), encoding="utf-8")


def create_test_client(monkeypatch, saved_jobs_dir: Path, queued_jobs_dir: Path, state_dir: Path | None = None):
    monkeypatch.setattr(routes_module, "SAVED_JOBS_DIR", str(saved_jobs_dir))
    monkeypatch.setattr(routes_module, "QUEUED_JOBS_DIR", str(queued_jobs_dir))
    if state_dir is None:
        state_dir = saved_jobs_dir.parent / "state"

    app = create_app(TestConfig)
    app.config["SAVED_JOBS_DIR"] = str(saved_jobs_dir)
    app.config["QUEUED_JOBS_DIR"] = str(queued_jobs_dir)
    app.config["TRACE_DIR"] = str(saved_jobs_dir)
    app.config["STATE_DIR"] = str(state_dir)
    return app.test_client()


def test_batch_routes_use_merged_saved_and_queued_claims(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        saved_jobs_dir,
        "batch-mixed",
        "claim-saved",
        status="processed",
        text="Saved claim",
        report_available=True,
    )
    write_claim(
        queued_jobs_dir,
        "batch-mixed",
        "claim-queued",
        status="ready_for_analysis",
        text="Queued claim",
        report_available=False,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    batch_response = client.get("/api/v1/batch/batch-mixed")
    assert batch_response.status_code == 200
    batch_payload = batch_response.get_json()
    assert batch_payload["status"] == "processing"
    assert batch_payload["total_claims"] == 2
    assert batch_payload["processed_claims"] == 1
    assert batch_payload["counts_by_location"] == {"saved_jobs": 1, "queued_jobs": 1}

    claims_by_id = {claim["claim_id"]: claim for claim in batch_payload["claims"]}
    assert claims_by_id["claim-saved"]["claim_location"] == "saved_jobs"
    assert claims_by_id["claim-saved"]["report_available"] is True
    assert claims_by_id["claim-queued"]["claim_location"] == "queued_jobs"
    assert claims_by_id["claim-queued"]["report_available"] is False

    progress_response = client.get("/api/v1/batch/batch-mixed/progress")
    assert progress_response.status_code == 200
    progress_payload = progress_response.get_json()
    assert progress_payload["status"] == "processing"
    assert progress_payload["total_claims"] == 2
    assert progress_payload["processed_claims"] == 1
    assert progress_payload["current_claim_id"] == "claim-queued"
    # No heartbeat file has been written, so the processor reports as down.
    assert progress_payload["processor"]["alive"] is False

    write_heartbeat(tmp_path / "state")
    progress_payload = client.get("/api/v1/batch/batch-mixed/progress").get_json()
    assert progress_payload["processor"]["alive"] is True
    assert progress_payload["processor"]["age_seconds"] is not None

    browse_response = client.get("/api/v1/browse")
    assert browse_response.status_code == 200
    browse_payload = browse_response.get_json()
    batch_entry = next(batch for batch in browse_payload["batches"] if batch["batch_id"] == "batch-mixed")
    assert batch_entry["status"] == "processing"
    assert batch_entry["counts_by_location"] == {"saved_jobs": 1, "queued_jobs": 1}
    preview_by_id = {claim["claim_id"]: claim for claim in batch_entry["preview_claims"]}
    assert preview_by_id["claim-queued"]["claim_location"] == "queued_jobs"


def test_duplicate_claim_prefers_queued_copy(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        saved_jobs_dir,
        "batch-dup",
        "claim-1",
        status="processed",
        text="Saved copy",
        report_available=True,
    )
    write_claim(
        queued_jobs_dir,
        "batch-dup",
        "claim-1",
        status="ready_for_search",
        text="Queued copy",
        report_available=False,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.get("/api/v1/batch/batch-dup")

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["claims"]) == 1
    claim = payload["claims"][0]
    assert claim["text"] == "Queued copy"
    assert claim["status"] == "ready_for_search"
    assert claim["claim_location"] == "queued_jobs"
    assert claim["report_available"] is False


def test_active_claim_report_endpoint_returns_processing(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        queued_jobs_dir,
        "batch-active",
        "claim-1",
        status="ready_for_analysis",
        text="Queued claim",
        report_available=False,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.get("/api/v1/claims/batch-active/claim-1/report")

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "CLAIM_PROCESSING"
    assert payload["claim_location"] == "queued_jobs"


def test_batch_results_accepts_queued_only_batch(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        queued_jobs_dir,
        "batch-queued-only",
        "claim-1",
        status="queued",
        text="Queued only claim",
        report_available=False,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.get("/batch_results?batch_id=batch-queued-only")

    assert response.status_code == 200


def test_download_batch_markdown_rejects_incomplete_batch(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        saved_jobs_dir,
        "batch-markdown",
        "claim-saved",
        status="processed",
        text="Saved claim",
        report_available=True,
    )
    write_claim(
        queued_jobs_dir,
        "batch-markdown",
        "claim-queued",
        status="ready_for_analysis",
        text="Queued claim",
        report_available=False,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.get("/api/v1/batch/batch-markdown/download_markdown")

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "BATCH_NOT_COMPLETED"


def test_download_batch_markdown_supports_claim_store_batch_view(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    state_dir = tmp_path / "state"

    claim_data = {
        "text": "Creatine improves memory in adults.",
        "status": "processed",
        "batch_id": "batch-store-markdown",
        "claim_id": "claim-store",
        "report": {
            "claimRating": 4,
            "explanation": "Evidence leans supportive.",
            "relevantPapers": [{"title": "Processed paper"}],
            "searchQueries": ["memory trial"],
        },
    }

    store = ClaimStore(
        state_dir=str(state_dir),
        saved_jobs_dir=str(saved_jobs_dir),
        queued_jobs_dir=str(queued_jobs_dir),
        trace_dir=str(saved_jobs_dir),
    )
    claim_record, _ = store.get_or_create_claim(
        claim_data["text"],
        batch_tags=["batch-store-markdown"],
    )
    store.create_run(
        claim_record=claim_record,
        batch_tags=["batch-store-markdown"],
        transport_batch_id="batch-store-markdown",
        transport_claim_id="claim-store",
        status="processed",
        initial_claim_data=claim_data,
    )

    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir, state_dir=state_dir)
    response = client.get("/api/v1/batch/batch-store-markdown/download_markdown")

    assert response.status_code == 200
    archive = zipfile.ZipFile(BytesIO(response.data))
    assert archive.namelist() == ["claim_claim-store.txt"]
    markdown = archive.read("claim_claim-store.txt").decode("utf-8")
    assert "Creatine improves memory in adults." in markdown
    assert "Evidence leans supportive." in markdown


def test_delete_batch_removes_saved_and_queued_dirs(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        saved_jobs_dir,
        "batch-delete",
        "claim-saved",
        status="processed",
        text="Saved claim",
        report_available=True,
    )
    write_claim(
        queued_jobs_dir,
        "batch-delete",
        "claim-queued",
        status="ready_for_search",
        text="Queued claim",
        report_available=False,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.delete("/api/v1/delete/batch/batch-delete")

    assert response.status_code == 200
    assert not (saved_jobs_dir / "batch-delete").exists()
    assert not (queued_jobs_dir / "batch-delete").exists()


def test_models_for_batch_resolves_from_both_state_shapes():
    from app.api.routes import _models_for_batch

    # File-derived shape: provider info nested under claim_data.
    file_shape = {
        "claims": [
            {
                "claim_data": {
                    "provider_snapshot": {"label": "Ollama", "default_model": "llama3.1:8b"},
                    "model_overrides": {},
                }
            }
        ]
    }
    assert _models_for_batch(file_shape) == [
        {"provider": "Ollama", "models": ["llama3.1:8b"], "display": "llama3.1:8b"}
    ]

    # Run-summary shape: provider info at the top level.
    summary_shape = {
        "claims": [
            {
                "provider_snapshot": {"label": "OpenAI", "default_model": "gpt-4o"},
                "model_overrides": {},
            }
        ]
    }
    assert _models_for_batch(summary_shape) == [
        {"provider": "OpenAI", "models": ["gpt-4o"], "display": "gpt-4o"}
    ]

    # Empty / missing state is safe.
    assert _models_for_batch(None) == []
    assert _models_for_batch({"claims": []}) == []


def test_models_for_batch_dedupes_and_lists_per_stage_overrides():
    from app.api.routes import _models_for_batch

    state = {
        "claims": [
            # Two claims on the same model collapse to one entry.
            {"provider_snapshot": {"label": "OpenAI", "default_model": "gpt-4o"}},
            {"provider_snapshot": {"label": "OpenAI", "default_model": "gpt-4o"}},
            # Per-stage overrides surface each distinct model.
            {
                "provider_snapshot": {"label": "OpenAI", "default_model": "gpt-4o"},
                "model_overrides": {"final_report": "gpt-5"},
            },
        ]
    }
    result = _models_for_batch(state)
    assert {"provider": "OpenAI", "models": ["gpt-4o"], "display": "gpt-4o"} in result
    override_entry = next(e for e in result if "gpt-5" in e["models"])
    assert override_entry["models"] == ["gpt-4o", "gpt-5"]
    assert len(result) == 2  # the two identical gpt-4o claims deduped


def test_batch_progress_endpoint_includes_models(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(
        queued_jobs_dir,
        "batch-models",
        "claim-1",
        status="ready_for_search",
        text="A claim being processed",
        report_available=False,
        provider_snapshot={"label": "Ollama", "default_model": "llama3.1:8b"},
        model_overrides={},
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    payload = client.get("/api/v1/batch/batch-models/progress").get_json()
    assert payload["models"] == [
        {"provider": "Ollama", "models": ["llama3.1:8b"], "display": "llama3.1:8b"}
    ]
