import json
import sys
import types
from pathlib import Path

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app import create_app
from app.api import routes as routes_module
from app.config.settings import Config


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


def write_claim_file(
    root_dir: Path,
    batch_id: str,
    claim_id: str,
    *,
    status: str,
    queries=None,
    raw_papers=None,
    report=None,
):
    batch_dir = root_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    claim_data = {
        "text": "Test claim",
        "status": status,
        "batch_id": batch_id,
        "claim_id": claim_id,
        "search_config": {"num_queries": 4, "results_per_query": 5},
        "semantic_scholar_queries": queries if queries is not None else ["memory trial"],
        "raw_papers": raw_papers if raw_papers is not None else [{"title": "Paper A", "corpusId": 1}],
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
    if report is not None:
        claim_data["report"] = report

    path = batch_dir / f"{claim_id}.txt"
    path.write_text(json.dumps(claim_data, indent=2), encoding="utf-8")
    return path


def write_trace_file(saved_jobs_dir: Path, batch_id: str, claim_id: str):
    trace_dir = saved_jobs_dir / batch_id / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_record = {
        "trace_id": f"trace-{claim_id}",
        "stage": "final_report",
        "status": "error",
        "latency_ms": 2500,
        "timeout_configured_s": 300,
        "timeout_source": "task_override",
        "timed_out": True,
    }
    (trace_dir / f"{claim_id}.jsonl").write_text(
        json.dumps(trace_record) + "\n",
        encoding="utf-8",
    )


def create_test_client(monkeypatch, saved_jobs_dir: Path, queued_jobs_dir: Path):
    monkeypatch.setattr(routes_module, "SAVED_JOBS_DIR", str(saved_jobs_dir))
    monkeypatch.setattr(routes_module, "QUEUED_JOBS_DIR", str(queued_jobs_dir))

    app = create_app(TestConfig)
    app.config["TRACE_DIR"] = str(saved_jobs_dir)
    return app.test_client()


def test_resume_final_stage_claim_requeues_to_analysis(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim_file(
        saved_jobs_dir,
        "batch-a",
        "claim-1",
        status="processed",
        queries=["query one"],
        raw_papers=[{"title": "Paper A", "corpusId": 1}],
        report={"explanation": "stale report"},
    )
    write_trace_file(saved_jobs_dir, "batch-a", "claim-1")
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.post("/api/v1/claims/batch-a/claim-1/resume")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["resume_to_status"] == "ready_for_analysis"
    saved_path = saved_jobs_dir / "batch-a" / "claim-1.txt"
    queued_path = queued_jobs_dir / "batch-a" / "claim-1.txt"
    assert not saved_path.exists()
    assert queued_path.exists()
    queued_data = json.loads(queued_path.read_text(encoding="utf-8"))
    assert queued_data["status"] == "ready_for_analysis"
    assert "report" not in queued_data
    assert (saved_jobs_dir / "batch-a" / "traces" / "claim-1.jsonl").exists()


def test_resume_infers_earlier_stages(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    write_claim_file(
        saved_jobs_dir,
        "batch-b",
        "claim-no-queries",
        status="processed",
        queries=[],
        raw_papers=[],
        report={"explanation": "failed before search"},
    )
    response = client.post("/api/v1/claims/batch-b/claim-no-queries/resume")
    assert response.status_code == 200
    assert response.get_json()["resume_to_status"] == "queued"

    write_claim_file(
        saved_jobs_dir,
        "batch-c",
        "claim-no-raw",
        status="processed",
        queries=["query one"],
        raw_papers=[],
        report={"explanation": "failed before paper fetch completed"},
    )
    response = client.post("/api/v1/claims/batch-c/claim-no-raw/resume")
    assert response.status_code == 200
    assert response.get_json()["resume_to_status"] == "ready_for_search"


def test_resume_returns_conflict_when_claim_already_queued(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim_file(
        queued_jobs_dir,
        "batch-d",
        "claim-queued",
        status="ready_for_analysis",
        report=None,
    )
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.post("/api/v1/claims/batch-d/claim-queued/resume")

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "CLAIM_ALREADY_QUEUED"


def test_trace_records_include_resume_metadata(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim_file(
        saved_jobs_dir,
        "batch-e",
        "claim-trace",
        status="processed",
        queries=["query one"],
        raw_papers=[{"title": "Paper A", "corpusId": 1}],
        report={"explanation": "failed at final report"},
    )
    write_trace_file(saved_jobs_dir, "batch-e", "claim-trace")
    client = create_test_client(monkeypatch, saved_jobs_dir, queued_jobs_dir)

    response = client.get("/api/v1/claims/batch-e/claim-trace/trace_records")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["claim_status"] == "processed"
    assert payload["claim_location"] == "saved_jobs"
    assert payload["resume_available"] is True
    assert payload["resume_stage"] == "ready_for_analysis"
    assert "Saved paper search output exists" in payload["resume_reason"]
    assert payload["records"][0]["timed_out"] is True
    assert payload["records"][0]["timeout_source"] == "task_override"
