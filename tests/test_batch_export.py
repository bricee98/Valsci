import json
import subprocess
import sys
import types
from pathlib import Path

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app import create_app
from app.api import routes as routes_module
from app.config.settings import Config


ROOT = Path(__file__).resolve().parents[1]


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


def write_claim(saved_jobs_dir: Path, batch_id: str, claim_id: str, text: str) -> None:
    batch_dir = saved_jobs_dir / batch_id
    (batch_dir / "traces").mkdir(parents=True, exist_ok=True)
    (batch_dir / "issues").mkdir(parents=True, exist_ok=True)

    claim_data = {
        "text": text,
        "status": "processed",
        "batch_id": batch_id,
        "claim_id": claim_id,
        "search_config": {"num_queries": 4, "results_per_query": 5},
        "semantic_scholar_queries": ["memory trial", "cognition adults"],
        "raw_papers": [{"title": "Candidate paper", "url": "https://example.org/candidate"}],
        "processed_papers": [
            {
                "paper": {
                    "title": "Processed paper",
                    "url": "https://example.org/processed",
                    "corpusId": 123,
                },
                "relevance": 0.9,
                "score": 0.8,
            }
        ],
        "non_relevant_papers": [],
        "inaccessible_papers": [],
        "report": {
            "claimRating": 4,
            "explanation": "Evidence leans supportive.",
            "searchQueries": ["memory trial", "cognition adults"],
            "relevantPapers": [{"title": "Processed paper", "link": "https://example.org/processed"}],
            "usage_summary": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cost_usd": 0.01,
                "is_estimated": False,
            },
            "issues": [
                {
                    "severity": "WARNING",
                    "stage": "paper_analysis",
                    "message": "One paper was skipped",
                }
            ],
            "debug_trace": {
                "summary": {
                    "llm_calls": 2,
                    "models_used": ["gpt-5"],
                    "retries": 0,
                    "context_overflow_prevented": 0,
                }
            },
        },
    }
    (batch_dir / f"{claim_id}.txt").write_text(
        json.dumps(claim_data, indent=2),
        encoding="utf-8",
    )
    (batch_dir / "traces" / f"{claim_id}.jsonl").write_text(
        json.dumps({"trace_id": f"trace-{claim_id}", "stage": "final_report"}) + "\n",
        encoding="utf-8",
    )
    (batch_dir / "issues" / f"{claim_id}.jsonl").write_text(
        json.dumps(
            {
                "issue_id": f"issue-{claim_id}",
                "severity": "WARNING",
                "stage": "paper_analysis",
                "message": "One paper was skipped",
            }
        ) + "\n",
        encoding="utf-8",
    )


def create_test_client(monkeypatch, saved_jobs_dir: Path):
    monkeypatch.setattr(routes_module, "SAVED_JOBS_DIR", str(saved_jobs_dir))
    monkeypatch.setattr(routes_module, "QUEUED_JOBS_DIR", str(saved_jobs_dir.parent / "queued_jobs"))

    app = create_app(TestConfig)
    app.config["TRACE_DIR"] = str(saved_jobs_dir)
    return app.test_client()


def test_export_batches_route_returns_attachment(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    write_claim(saved_jobs_dir, "batch-one", "claim-a", "Creatine improves memory in adults.")
    write_claim(saved_jobs_dir, "batch-two", "claim-b", "Vitamin D reduces falls in older adults.")

    client = create_test_client(monkeypatch, saved_jobs_dir)
    response = client.post(
        "/api/v1/batches/export",
        json={"batch_ids": ["batch-one", "batch-two"]},
    )

    assert response.status_code == 200
    assert "attachment;" in response.headers["Content-Disposition"]

    payload = json.loads(response.data)
    assert payload["batch_ids"] == ["batch-one", "batch-two"]
    assert len(payload["batches"]) == 2
    first_claim = payload["batches"][0]["claims"][0]
    assert first_claim["claim_data"]["text"] == "Creatine improves memory in adults."
    assert first_claim["artifacts"]["trace_records"][0]["trace_id"] == "trace-claim-a"
    assert first_claim["artifacts"]["issue_records"][0]["issue_id"] == "issue-claim-a"


def test_export_batches_route_requires_selected_batches(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    saved_jobs_dir.mkdir(parents=True, exist_ok=True)
    client = create_test_client(monkeypatch, saved_jobs_dir)

    response = client.post("/api/v1/batches/export", json={"batch_ids": []})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["code"] == "NO_BATCH_IDS"


def test_export_batches_route_includes_queued_claims(monkeypatch, tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    queued_jobs_dir = tmp_path / "queued_jobs"
    write_claim(saved_jobs_dir, "batch-mixed", "claim-saved", "Completed claim")
    write_claim(queued_jobs_dir, "batch-mixed", "claim-queued", "Resumed claim still processing")

    queued_claim_path = queued_jobs_dir / "batch-mixed" / "claim-queued.txt"
    queued_payload = json.loads(queued_claim_path.read_text(encoding="utf-8"))
    queued_payload["status"] = "ready_for_analysis"
    queued_payload.pop("report", None)
    queued_claim_path.write_text(json.dumps(queued_payload, indent=2), encoding="utf-8")

    client = create_test_client(monkeypatch, saved_jobs_dir)
    response = client.post(
        "/api/v1/batches/export",
        json={"batch_ids": ["batch-mixed"]},
    )

    assert response.status_code == 200
    payload = json.loads(response.data)
    claims = payload["batches"][0]["claims"]
    claims_by_id = {claim["claim_id"]: claim for claim in claims}
    assert set(claims_by_id) == {"claim-saved", "claim-queued"}
    assert claims_by_id["claim-saved"]["claim_location"] == "saved_jobs"
    assert claims_by_id["claim-saved"]["report_available"] is True
    assert claims_by_id["claim-queued"]["claim_location"] == "queued_jobs"
    assert claims_by_id["claim-queued"]["report_available"] is False


def test_export_batches_cli_can_select_batches_by_claim_regex(tmp_path):
    saved_jobs_dir = tmp_path / "saved_jobs"
    write_claim(saved_jobs_dir, "batch-memory", "claim-a", "Creatine improves memory in adults.")
    write_claim(saved_jobs_dir, "batch-bone", "claim-b", "Vitamin D reduces falls in older adults.")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_batches.py"),
        "--saved-jobs-dir",
        str(saved_jobs_dir),
        "--claim-regex",
        "memory",
        "--ignore-case",
        "--format",
        "json",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["batch_ids"] == ["batch-memory"]
    assert len(payload["batches"]) == 1
    assert payload["batches"][0]["claims"][0]["claim_data"]["claim_id"] == "claim-a"
