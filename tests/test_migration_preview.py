import json
import sys
import types
from pathlib import Path


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app.services.claim_store import ClaimStore


def _store(tmp_path: Path) -> ClaimStore:
    return ClaimStore(
        state_dir=str(tmp_path / "state"),
        saved_jobs_dir=str(tmp_path / "saved_jobs"),
        queued_jobs_dir=str(tmp_path / "queued_jobs"),
        trace_dir=str(tmp_path / "saved_jobs"),
    )


def _write_claim(root: Path, batch_id: str, claim_id: str, data: dict):
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / f"{claim_id}.txt").write_text(json.dumps(data), encoding="utf-8")


def test_preview_legacy_report_with_report(tmp_path):
    store = _store(tmp_path)
    _write_claim(
        tmp_path / "saved_jobs",
        "batch-1",
        "claim-1",
        {
            "text": "PAH variants cause PKU.",
            "status": "processed",
            "report": {
                "claimRating": 4,
                "explanation": "Most evidence supports the claim.",
                "relevantPapers": [{"title": "A"}, {"title": "B"}, {"title": "C"}],
                "nonRelevantPapers": [{"title": "D"}],
                "inaccessiblePapers": [{"title": "E"}, {"title": "F"}],
                "searchQueries": ["q1", "q2"],
            },
        },
    )

    preview = store.preview_legacy_report("batch-1", "claim-1", root="saved_jobs")
    assert preview is not None
    assert preview["source_root"] == "saved_jobs"
    assert preview["has_report"] is True
    rp = preview["report_preview"]
    assert rp["rating"] == 4
    assert rp["explanation"].startswith("Most evidence")
    assert rp["evidence"] == {
        "relevant": 3,
        "non_relevant": 1,
        "inaccessible": 2,
        "search_queries": 2,
        "total_papers": 6,
    }


def test_preview_legacy_report_no_report(tmp_path):
    store = _store(tmp_path)
    _write_claim(
        tmp_path / "saved_jobs",
        "batch-1",
        "claim-empty",
        {"text": "No report here", "status": "processed"},
    )
    preview = store.preview_legacy_report("batch-1", "claim-empty")
    assert preview is not None
    assert preview["has_report"] is False
    assert preview["report_preview"] is None


def test_preview_legacy_report_queued_claim(tmp_path):
    store = _store(tmp_path)
    _write_claim(
        tmp_path / "queued_jobs",
        "batch-q",
        "claim-q",
        {"text": "Still processing", "status": "processing"},
    )
    preview = store.preview_legacy_report("batch-q", "claim-q", root="queued_jobs")
    assert preview is not None
    assert preview["source_root"] == "queued_jobs"
    assert preview["status"] == "processing"
    assert preview["has_report"] is False


def test_preview_legacy_report_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.preview_legacy_report("nope", "nope") is None


def test_preview_legacy_report_invalid_root_raises(tmp_path):
    store = _store(tmp_path)
    try:
        store.preview_legacy_report("b", "c", root="bogus")
    except ValueError as exc:
        assert "saved_jobs" in str(exc)
    else:
        raise AssertionError("invalid root must raise ValueError")


def test_migration_report_route(tmp_path, monkeypatch):
    from app import create_app
    from app.api import routes as routes_module
    from app.config.settings import Config

    class TestConfig(Config):
        TESTING = True
        REQUIRE_PASSWORD = False

    store = _store(tmp_path)
    _write_claim(
        tmp_path / "saved_jobs",
        "batch-1",
        "claim-1",
        {
            "text": "Routed claim",
            "status": "processed",
            "report": {"claimRating": 2, "explanation": "Mixed.", "relevantPapers": []},
        },
    )
    monkeypatch.setattr(routes_module, "_claim_store", lambda: store)

    app = create_app(TestConfig)
    client = app.test_client()

    ok = client.get("/api/v1/migration/batches/batch-1/claims/claim-1/report?root=saved_jobs")
    assert ok.status_code == 200
    payload = ok.get_json()
    assert payload["has_report"] is True
    assert payload["report_preview"]["rating"] == 2

    bad_root = client.get("/api/v1/migration/batches/batch-1/claims/claim-1/report?root=elsewhere")
    assert bad_root.status_code == 400

    missing = client.get("/api/v1/migration/batches/batch-1/claims/does-not-exist/report")
    assert missing.status_code == 404
