import json
from pathlib import Path

import pytest

from scripts import release_integrity as ri


def _make_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "semantic_scholar" / "manifests").mkdir(parents=True)
    (dist / "valsci-image.tar.gz").write_bytes(b"fake-image-bytes")
    (dist / "requirements.lock").write_text("flask==1.0 \\\n    --hash=sha256:abc\n", encoding="utf-8")
    (dist / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (dist / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (dist / "semantic_scholar" / "manifests" / "mendelian_v1.json").write_text(
        json.dumps({"release_id": "2026-05-26"}), encoding="utf-8"
    )
    return dist


_FILES = [
    "valsci-image.tar.gz",
    "requirements.lock",
    "docker-compose.yml",
    "Dockerfile",
    "semantic_scholar/manifests",
]


def test_generate_creates_sums_and_manifest(tmp_path):
    dist = _make_dist(tmp_path)
    manifest = ri.generate(dist, _FILES, version="v1", created="2026-06-05T00:00:00Z")

    assert (dist / "SHA256SUMS").exists()
    assert (dist / "release_manifest.json").exists()
    assert manifest["version"] == "v1"
    assert manifest["algorithm"] == "sha256"

    paths = {entry["path"] for entry in manifest["files"]}
    assert "valsci-image.tar.gz" in paths
    assert "requirements.lock" in paths
    assert "docker-compose.yml" in paths
    assert "Dockerfile" in paths
    assert "semantic_scholar/manifests/mendelian_v1.json" in paths
    # The generated artifacts never hash themselves.
    assert "SHA256SUMS" not in paths
    assert "release_manifest.json" not in paths

    # SHA256SUMS lines are sha256sum-compatible: "<hex>  <relpath>".
    lines = [l for l in (dist / "SHA256SUMS").read_text().splitlines() if l.strip()]
    assert len(lines) == manifest["file_count"]
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert len(digest) == 64


def test_verify_passes_for_untouched_distribution(tmp_path):
    dist = _make_dist(tmp_path)
    ri.generate(dist, _FILES)
    ok, errors = ri.verify(dist)
    assert ok is True
    assert errors == []


def test_verify_fails_on_tampered_artifact(tmp_path):
    dist = _make_dist(tmp_path)
    ri.generate(dist, _FILES)
    (dist / "valsci-image.tar.gz").write_bytes(b"tampered!!")
    ok, errors = ri.verify(dist)
    assert ok is False
    assert any("valsci-image.tar.gz" in e and "mismatch" in e for e in errors)


def test_verify_fails_on_missing_artifact(tmp_path):
    dist = _make_dist(tmp_path)
    ri.generate(dist, _FILES)
    (dist / "Dockerfile").unlink()
    ok, errors = ri.verify(dist)
    assert ok is False
    assert any("Dockerfile" in e and "missing" in e for e in errors)


def test_verify_fails_on_tampered_manifest_metadata(tmp_path):
    dist = _make_dist(tmp_path)
    ri.generate(dist, _FILES)
    manifest_path = dist / "release_manifest.json"
    data = json.loads(manifest_path.read_text())
    data["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    ok, errors = ri.verify(dist)
    assert ok is False


def test_verify_fails_when_metadata_absent(tmp_path):
    dist = _make_dist(tmp_path)
    ok, errors = ri.verify(dist)
    assert ok is False
    assert any("SHA256SUMS" in e for e in errors)


def test_cli_generate_and_verify(tmp_path):
    dist = _make_dist(tmp_path)
    rc = ri.main(["generate", "--dist", str(dist), *_FILES])
    assert rc == 0
    rc = ri.main(["verify", "--dist", str(dist)])
    assert rc == 0
    (dist / "requirements.lock").write_text("changed", encoding="utf-8")
    rc = ri.main(["verify", "--dist", str(dist)])
    assert rc == 1
