import sys
import types
import json
from pathlib import Path


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app import create_app
from app.api import routes as routes_module
from app.config.settings import Config
from app.services import env_config
from app.services.data_manager import DataJobManager, RELEASE_ID_PATTERN, _mini_manifest_coverage


class TestConfig(Config):
    TESTING = True
    REQUIRE_PASSWORD = False


class FakeDataJobManager:
    def active_job(self):
        return None

    def list_jobs(self):
        return []


def _fake_data_state(tmp_path: Path):
    return {
        "base_dir": str(tmp_path / "semantic_scholar" / "datasets"),
        "index_dir": str(tmp_path / "semantic_scholar" / "datasets" / "binary_indices"),
        "api_key_present": False,
        "latest_release": None,
        "active_release": None,
        "releases": [],
        "dataset_options": [
            {"name": "papers", "label": "Papers", "note": "Paper metadata.", "default": True},
            {"name": "s2orc_v2", "label": "S2ORC v2", "note": "Preferred full text.", "default": True},
        ],
        "mini_manifest": {
            "manifest": "mendelian_v1.json",
            "path": str(tmp_path / "semantic_scholar" / "manifests" / "mendelian_v1.json"),
            "exists": False,
            "hash": None,
            "error": None,
        },
    }


def test_data_page_and_status_api(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_module, "build_data_state", lambda: _fake_data_state(tmp_path))
    monkeypatch.setattr(routes_module, "data_job_manager", lambda: FakeDataJobManager())
    monkeypatch.setattr(Config, "STATE_DIR", str(tmp_path / "state"), raising=False)

    app = create_app(TestConfig)
    app.config["STATE_DIR"] = str(tmp_path / "state")
    client = app.test_client()

    page = client.get("/data")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Local Data State" in html
    assert "data.js" in html
    assert ">Data</a>" in html

    response = client.get("/api/v1/data/status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"]["dataset_options"][1]["name"] == "s2orc_v2"
    assert payload["active_job"] is None


def test_new_release_type_picker_has_scoped_radio_layout():
    html = Path("app/templates/data.html").read_text(encoding="utf-8")
    css = Path("app/static/overhaul.css").read_text(encoding="utf-8")

    assert "release-type-options" in html
    assert "release-type-option" in html
    assert "segmented-control" not in html
    assert ".release-type-options" in css
    assert ".release-type-option" in css


def test_home_data_card_uses_release_readiness_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "STATE_DIR", str(tmp_path / "state"), raising=False)

    app = create_app(TestConfig)
    app.config["STATE_DIR"] = str(tmp_path / "state")
    client = app.test_client()

    page = client.get("/")

    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Current Semantic Scholar release and index readiness." in html
    assert "Local data" not in html
    assert 'id="homeDataSummary"' in html


def test_settings_page_and_env_api(monkeypatch, tmp_path):
    env_path = tmp_path / "env_vars.json"
    env_path.write_text(
        json.dumps(
            {
                "FLASK_SECRET_KEY": "dev-secret",
                "USER_EMAIL": "dev@example.com",
                "SEMANTIC_SCHOLAR_API_KEY": "",
                "LLM_PROVIDER": "ollama",
                "LLM_BASE_URL": "http://localhost:11434/v1",
                "REQUIRE_PASSWORD": "false",
                "SMTP_PORT": "587",
                "LLM_ROUTING": "{\"enabled\": false, \"tasks\": {\"query_generation\": {\"max_output_tokens\": 16000}}}",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(env_config.settings_module, "env_file_path", env_path)
    monkeypatch.setattr(Config, "STATE_DIR", str(tmp_path / "state"), raising=False)
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_API_KEY", "", raising=False)

    app = create_app(TestConfig)
    app.config["STATE_DIR"] = str(tmp_path / "state")
    client = app.test_client()

    page = client.get("/settings")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Settings" in html
    assert "settings.js" in html

    response = client.get("/api/v1/settings/env")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["path"] == str(env_path)
    entries = {entry["env_key"]: entry for entry in payload["entries"]}
    assert "SEMANTIC_SCHOLAR_API_KEY" in entries
    assert entries["REQUIRE_PASSWORD"]["value"] is False
    assert entries["SMTP_PORT"]["value"] == 587
    assert entries["LLM_ROUTING"]["value"]["enabled"] is False

    # Enriched per-entry metadata for the grouped UI.
    assert entries["LLM_ROUTING"]["category"] == "routing"
    # Gateway-backed settings hot-reload (no restart); startup-captured ones don't.
    assert entries["LLM_ROUTING"]["restart_required"] is False
    assert entries["LLM_ROUTING"]["label"] and entries["LLM_ROUTING"]["description"]
    assert entries["SMTP_PORT"]["restart_required"] is True
    assert entries["RATE_LIMIT_MAX_TOKENS_PER_CLAIM"]["restart_required"] is True

    # Processor config-sync status is attached for the live UI indicator.
    assert "processor" in payload
    assert set(payload["processor"]) >= {"alive", "config_synced", "env_file_mtime"}

    # Grouped structure + the friendly per-task output budget surfacing.
    group_ids = {group["id"] for group in payload["groups"]}
    assert {"provider", "routing", "performance"} <= group_ids
    routing_group = next(group for group in payload["groups"] if group["id"] == "routing")
    assert any(entry["env_key"] == "LLM_ROUTING" for entry in routing_group["entries"])
    assert payload["routing_output_budgets"]["query_generation"] == 16000
    assert payload["routing_output_budgets"]["final_report"] is None
    assert [stage["key"] for stage in payload["routing_task_stages"]] == [
        "query_generation", "paper_analysis", "venue_scoring", "final_report",
    ]

    response = client.put(
        "/api/v1/settings/env",
        json={"updates": {"SEMANTIC_SCHOLAR_API_KEY": "test-s2-key"}},
    )
    assert response.status_code == 200
    updated_file = json.loads(env_path.read_text(encoding="utf-8"))
    assert updated_file["SEMANTIC_SCHOLAR_API_KEY"] == "test-s2-key"
    assert Config.SEMANTIC_SCHOLAR_API_KEY == "test-s2-key"
    assert app.config["SEMANTIC_SCHOLAR_API_KEY"] == "test-s2-key"


def test_processor_config_status_reflects_heartbeat(monkeypatch, tmp_path):
    from app.services.processor_heartbeat import write_heartbeat

    env_path = tmp_path / "env_vars.json"
    env_path.write_text(json.dumps({"LLM_PROVIDER": "ollama"}), encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(env_config.settings_module, "env_file_path", env_path)
    monkeypatch.setattr(Config, "STATE_DIR", str(state_dir), raising=False)

    app = create_app(TestConfig)
    app.config["STATE_DIR"] = str(state_dir)
    client = app.test_client()

    # No heartbeat: processor down, not synced.
    down = client.get("/api/v1/settings/processor-status").get_json()
    assert down["alive"] is False
    assert down["config_synced"] is False

    # Heartbeat stamped with a config mtime at/after the env file: synced.
    env_mtime = env_path.stat().st_mtime
    write_heartbeat(state_dir, config_mtime=env_mtime)
    synced = client.get("/api/v1/settings/processor-status").get_json()
    assert synced["alive"] is True
    assert synced["config_synced"] is True

    # Heartbeat stamped before a newer save: alive but not yet synced.
    write_heartbeat(state_dir, config_mtime=env_mtime - 100)
    pending = client.get("/api/v1/settings/processor-status").get_json()
    assert pending["alive"] is True
    assert pending["config_synced"] is False


def test_routing_output_budgets_extraction():
    extract = env_config.routing_output_budgets
    # Full set, partial, and missing/invalid shapes all resolve cleanly.
    assert extract({"tasks": {"query_generation": {"max_output_tokens": 16000}}}) == {
        "query_generation": 16000,
        "paper_analysis": None,
        "venue_scoring": None,
        "final_report": None,
    }
    assert extract({}) == {s["key"]: None for s in env_config.ROUTING_TASK_STAGES}
    assert extract(None) == {s["key"]: None for s in env_config.ROUTING_TASK_STAGES}
    # Non-int values are ignored rather than surfaced as budgets.
    assert extract({"tasks": {"venue_scoring": {"max_output_tokens": "lots"}}})["venue_scoring"] is None
    assert extract("not-an-object") == {s["key"]: None for s in env_config.ROUTING_TASK_STAGES}


def test_settings_catalog_covers_every_known_config_key():
    # Every catalogued key maps to a real category, so nothing lands in a
    # mislabeled or non-existent group.
    category_ids = {category["id"] for category in env_config.SETTING_CATEGORIES}
    for env_key, info in env_config._SETTING_CATALOG.items():
        assert info["category"] in category_ids, f"{env_key} -> unknown category {info['category']}"
        assert info.get("label"), f"{env_key} missing label"


def test_data_warning_links_settings_and_page_titles_are_left_aligned():
    data_js = Path("app/static/data.js").read_text(encoding="utf-8")
    overhaul_css = Path("app/static/overhaul.css").read_text(encoding="utf-8")

    assert "/settings#SEMANTIC_SCHOLAR_API_KEY" in data_js
    assert ".page-title" in overhaul_css
    assert "text-align: left" in overhaul_css


def test_guidebook_documents_data_tab_and_markdown_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "STATE_DIR", str(tmp_path / "state"), raising=False)

    app = create_app(TestConfig)
    app.config["STATE_DIR"] = str(tmp_path / "state")
    client = app.test_client()

    page = client.get("/guidebook")

    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Copy as Markdown" in html
    assert 'id="data"' in html
    assert "Semantic Scholar Readiness" in html
    assert "Dataset Coverage Panel" in html
    assert "Current Job Panel" in html


def test_data_job_manager_builds_dataset_commands(tmp_path):
    manager = DataJobManager(state_dir=tmp_path / "state", project_root=tmp_path)

    command = manager.build_command(
        "full",
        {"release": "2026-05-26", "datasets": ["papers", "s2orc_v2"]},
    )

    assert command[:3][-2:] == ["-u", "-m"] or "-m" in command
    assert "semantic_scholar.utils.downloader" in command
    dataset_index = command.index("--datasets")
    assert command[dataset_index:] == ["--datasets", "papers", "s2orc_v2"]
    assert "--release" in command

    mini_command = manager.build_command(
        "mini",
        {"manifest": "mendelian_v1.json"},
    )

    assert "--mini" in mini_command
    assert "--mini-manifest" in mini_command
    # The manifest is passed as a bare filename, never a path.
    assert mini_command[mini_command.index("--mini-manifest") + 1] == "mendelian_v1.json"

    full_verify_command = manager.build_command(
        "verify",
        {
            "release": "2026-05-26",
            "datasets": ["papers", "s2orc_v2"],
            "manifest": "mendelian_v1.json",
        },
    )

    assert "--verify" in full_verify_command
    assert "--mini" not in full_verify_command
    assert "--mini-manifest" not in full_verify_command
    assert full_verify_command[full_verify_command.index("--datasets"):] == [
        "--datasets",
        "papers",
        "s2orc_v2",
    ]

    mini_verify_command = manager.build_command(
        "verify",
        {
            "release": "2026-05-26-mini-mendelian-v1",
            "mini": True,
            "manifest": "mendelian_v1.json",
        },
    )

    assert "--verify" in mini_verify_command
    assert "--mini" in mini_verify_command
    assert "--mini-manifest" in mini_verify_command
    assert "--release" not in mini_verify_command

    # A path-like manifest override is rejected loudly at command-build time.
    try:
        manager.build_command("mini", {"manifest": "../escape.json"})
    except ValueError as exc:
        assert "plain filename" in str(exc)
    else:
        raise AssertionError("path-like manifest override must be rejected")

    try:
        manager.build_command("first_shard", {"datasets": ["papers"]})
    except ValueError as exc:
        assert "Unsupported data operation" in str(exc)
    else:
        raise AssertionError("first_shard must not be a supported data operation")


def test_mini_manifest_coverage_flags_stale_release():
    manifest_summary = {
        "mini_release_id": "2026-05-26-mini-mendelian-v1",
        "dataset_id_counts": {"papers": 595, "s2orc_v2": 100},
    }

    stale = _mini_manifest_coverage(
        release_id="2026-05-26-mini-mendelian-v1",
        records_written={"papers": 500, "s2orc_v2": 5},
        manifest_summary=manifest_summary,
    )
    ready = _mini_manifest_coverage(
        release_id="2026-05-26-mini-mendelian-v1",
        records_written={"papers": 595, "s2orc_v2": 100},
        manifest_summary=manifest_summary,
    )
    unrelated = _mini_manifest_coverage(
        release_id="2026-05-26",
        records_written={},
        manifest_summary=manifest_summary,
    )

    assert stale["state"] == "stale"
    assert stale["missing"] == {"papers": 95, "s2orc_v2": 95}
    assert ready["state"] == "ready"
    assert unrelated is None


def test_release_id_pattern_excludes_mini_workspace():
    assert RELEASE_ID_PATTERN.match("2026-05-26-mini-mendelian-v1")
    assert RELEASE_ID_PATTERN.match("2026-05-26")
    assert not RELEASE_ID_PATTERN.match("mini")
    assert not RELEASE_ID_PATTERN.match("binary_indices")


def test_data_page_reindex_button_and_confirmation():
    data_js = Path("app/static/data.js").read_text(encoding="utf-8")
    # The maintenance action is renamed to "Re-index" (no conditional Index/Reindex label).
    assert '>Re-index</button>' in data_js
    assert 'indexStatus.state === "ready" ? "Reindex"' not in data_js
    # Re-index requires a confirmation that states data files are unchanged.
    assert "window.confirm(" in data_js
    assert "Dataset files are not changed" in data_js
    assert "only the binary indices are rebuilt" in data_js


def test_data_page_persists_dataset_checkbox_selection_per_release():
    data_js = Path("app/static/data.js").read_text(encoding="utf-8")
    assert "releaseDatasetSelection" in data_js
    assert "ensureReleaseSelection" in data_js
    # Selection is keyed by release and reset when the selected release changes.
    assert "releaseDatasetSelection.releaseId !== release.release_id" in data_js
    assert "releaseDatasetSelection = null;" in data_js
    # A change listener keeps the stored selection in sync across re-renders.
    assert 'addEventListener("change"' in data_js


def test_data_page_surfaces_exit_code_and_stderr():
    data_js = Path("app/static/data.js").read_text(encoding="utf-8")
    assert "Exit Code" in data_js
    assert "stderr_tail" in data_js
    assert 'entry.stream === "stderr"' in data_js


def test_data_page_recent_jobs_have_view_log_button():
    data_js = Path("app/static/data.js").read_text(encoding="utf-8")
    # A "View log" button per recent job loads that job's full log into the panel.
    assert "data-view-log" in data_js
    assert "View log" in data_js
    assert "viewJobLog" in data_js
    # The Current Job panel keeps the most recent job visible when idle, so a
    # finished job's log (e.g. a failure) does not vanish on completion.
    assert "jobToDisplay" in data_js
    assert "selectedJobId" in data_js


def test_migration_page_scroll_and_report_preview_markup():
    migration_js = Path("app/static/migration.js").read_text(encoding="utf-8")
    # After "Review contents" the review panel scrolls into view.
    assert "scrollIntoView" in migration_js
    # Claims with a report expose an inline "Preview report" action.
    assert "Preview report" in migration_js
    assert "data-preview-claim" in migration_js
    assert "previewReport" in migration_js
    assert "report_preview" in migration_js
