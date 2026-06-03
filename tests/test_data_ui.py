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
            "path": str(tmp_path / "semantic_scholar" / "mini_corpora" / "mendelian_v1" / "manifest.json"),
            "exists": False,
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
                "LLM_PROVIDER": "local",
                "LLM_BASE_URL": "http://localhost:11434/v1",
                "REQUIRE_PASSWORD": "false",
                "SMTP_PORT": "587",
                "LLM_ROUTING": "{\"enabled\": false}",
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
    assert "Environment Values" in html
    assert "settings.js" in html

    response = client.get("/api/v1/settings/env")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["path"] == str(env_path)
    entries = {entry["env_key"]: entry for entry in payload["entries"]}
    assert "SEMANTIC_SCHOLAR_API_KEY" in entries
    assert entries["REQUIRE_PASSWORD"]["value"] is False
    assert entries["SMTP_PORT"]["value"] == 587
    assert entries["LLM_ROUTING"]["value"] == {"enabled": False}

    response = client.put(
        "/api/v1/settings/env",
        json={"updates": {"SEMANTIC_SCHOLAR_API_KEY": "test-s2-key"}},
    )
    assert response.status_code == 200
    updated_file = json.loads(env_path.read_text(encoding="utf-8"))
    assert updated_file["SEMANTIC_SCHOLAR_API_KEY"] == "test-s2-key"
    assert Config.SEMANTIC_SCHOLAR_API_KEY == "test-s2-key"
    assert app.config["SEMANTIC_SCHOLAR_API_KEY"] == "test-s2-key"


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
        {"manifest_path": str(tmp_path / "mendelian_v1.json")},
    )

    assert "--mini" in mini_command
    assert "--mini-manifest" in mini_command

    full_verify_command = manager.build_command(
        "verify",
        {
            "release": "2026-05-26",
            "datasets": ["papers", "s2orc_v2"],
            "manifest_path": str(tmp_path / "mendelian_v1.json"),
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
            "manifest_path": str(tmp_path / "mendelian_v1.json"),
        },
    )

    assert "--verify" in mini_verify_command
    assert "--mini" in mini_verify_command
    assert "--mini-manifest" in mini_verify_command
    assert "--release" not in mini_verify_command

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
