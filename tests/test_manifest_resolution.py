import json
import sys
import types
from pathlib import Path

import pytest


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from app.config.settings import Config
from semantic_scholar.utils import downloader as downloader_module
from semantic_scholar.utils.downloader import (
    MANIFESTS_DIR,
    MiniCorpusManifestError,
    S2DatasetDownloader,
    configured_manifest_name,
    resolve_manifest_name,
    resolve_manifest_path,
)

ROOT = Path(__file__).resolve().parents[1]


def _bare_downloader() -> S2DatasetDownloader:
    return S2DatasetDownloader.__new__(S2DatasetDownloader)


def test_config_default_manifest_filename():
    assert Config.SEMANTIC_SCHOLAR_MANIFEST == "mendelian_v1.json"
    assert configured_manifest_name() == "mendelian_v1.json"


def test_default_manifest_resolves_inside_manifests_dir_and_exists():
    path = resolve_manifest_path()
    assert path == MANIFESTS_DIR / "mendelian_v1.json"
    assert path.parent == MANIFESTS_DIR
    assert path.exists(), "tracked default manifest must exist at the new location"


def test_resolve_manifest_name_accepts_plain_filename():
    assert resolve_manifest_name("mendelian_v1.json") == "mendelian_v1.json"
    assert resolve_manifest_name(None) == "mendelian_v1.json"
    assert resolve_manifest_name("   ") == "mendelian_v1.json"


@pytest.mark.parametrize(
    "value",
    [
        "../escape.json",
        "sub/dir.json",
        "/abs/path.json",
        "dir\\file.json",
        "..",
        ".",
    ],
)
def test_resolve_manifest_name_rejects_paths(value):
    with pytest.raises(MiniCorpusManifestError) as exc:
        resolve_manifest_name(value)
    assert "plain filename" in str(exc.value)


def test_configured_manifest_override(monkeypatch):
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_MANIFEST", "custom.json", raising=False)
    assert configured_manifest_name() == "custom.json"
    assert resolve_manifest_path() == MANIFESTS_DIR / "custom.json"


def test_configured_manifest_blank_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_MANIFEST", "", raising=False)
    assert configured_manifest_name() == "mendelian_v1.json"


def test_load_missing_manifest_raises(monkeypatch):
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_MANIFEST", "definitely_missing.json", raising=False)
    downloader = _bare_downloader()
    with pytest.raises(MiniCorpusManifestError) as exc:
        downloader._load_mini_manifest()
    assert "Curated mini-corpus manifest not found" in str(exc.value)


def test_load_invalid_json_manifest_raises(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not valid json", encoding="utf-8")
    downloader = _bare_downloader()
    with pytest.raises(MiniCorpusManifestError) as exc:
        downloader._load_mini_manifest(bad)
    assert "not valid JSON" in str(exc.value)


def test_load_manifest_missing_release_id_raises(tmp_path):
    bad = tmp_path / "no_release.json"
    bad.write_text(json.dumps({"datasets": {"papers": {"corpus_ids": [1]}}}), encoding="utf-8")
    downloader = _bare_downloader()
    with pytest.raises(MiniCorpusManifestError) as exc:
        downloader._load_mini_manifest(bad)
    assert "release_id" in str(exc.value)


def test_load_manifest_without_dataset_ids_raises(tmp_path):
    bad = tmp_path / "no_ids.json"
    bad.write_text(json.dumps({"release_id": "2026-05-26", "datasets": {}}), encoding="utf-8")
    downloader = _bare_downloader()
    with pytest.raises(MiniCorpusManifestError) as exc:
        downloader._load_mini_manifest(bad)
    assert "corpus_ids or author_ids" in str(exc.value)


def test_tracked_manifest_lives_only_in_manifests_dir():
    assert (ROOT / "semantic_scholar" / "manifests" / "mendelian_v1.json").exists()
    assert not (ROOT / "semantic_scholar" / "mini_corpora" / "mendelian_v1" / "manifest.json").exists()


def test_env_example_includes_manifest_setting():
    env_example = json.loads((ROOT / "env_vars.json.example").read_text(encoding="utf-8"))
    assert env_example.get("SEMANTIC_SCHOLAR_MANIFEST") == "mendelian_v1.json"


def test_docker_compose_mounts_manifests_readonly_for_both_services():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    mount = "./semantic_scholar/manifests:/valsci/semantic_scholar/manifests:ro"
    # Once for web, once for processor.
    assert compose.count(mount) == 2


def test_dockerfile_installs_from_hashed_lockfile():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements.lock" in dockerfile
    assert "--require-hashes" in dockerfile


def test_requirements_lock_exists_and_is_hash_pinned():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert "--hash=sha256:" in lock
    # Every top-level/transitive pin should carry at least one hash.
    pinned = [line for line in lock.splitlines() if "==" in line and not line.strip().startswith("#")]
    assert pinned, "lockfile must pin package versions"


def test_distribution_script_bundles_manifests_and_integrity():
    script = (ROOT / "create_distribution.sh").read_text(encoding="utf-8")
    assert "semantic_scholar/manifests" in script
    assert "release_integrity" in script
    assert "requirements.lock" in script
