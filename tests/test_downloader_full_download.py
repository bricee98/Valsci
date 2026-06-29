"""The full-corpus download now uses the same hardened streaming transfer as the
mini build (no wget): make_request + _ResumableHTTPReader via _iter_source_lines,
with re-signing on expired presigned URLs. These tests cover download_file's
streaming/skip/resign behavior and download_dataset's re-sign wiring."""
import json
import sys
import types

import pytest
import requests

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils.downloader import S2DatasetDownloader


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)


def _bare():
    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    d.session = types.SimpleNamespace(headers={})
    d.rate_limiter = types.SimpleNamespace(wait=lambda: None)
    return d


def _raising_iter(url):
    raise requests.exceptions.HTTPError("403 Forbidden", response=_Resp(403))
    yield  # pragma: no cover - marks this a generator


def test_download_file_streams_decompressed_lines_to_disk(tmp_path, monkeypatch):
    d = _bare()
    lines = [b'{"corpusid": 1}\n', b'{"corpusid": 2}\n']
    monkeypatch.setattr(d, "_iter_source_lines", lambda url: iter(lines))
    ok, path = d.download_file("https://h/p/shard.gz", tmp_path)
    assert ok is True
    assert path == tmp_path / "shard.json"          # .gz mapped to decompressed .json
    assert path.read_bytes() == b"".join(lines)     # byte-faithful reconstruction
    assert not (tmp_path / "shard.json.partial").exists()  # atomic: no partial left


def test_download_file_skips_when_already_present(tmp_path, monkeypatch):
    d = _bare()
    (tmp_path / "shard.json").write_text("existing", encoding="utf-8")
    called = {"n": 0}

    def iter_fn(url):
        called["n"] += 1
        yield b"{}\n"

    monkeypatch.setattr(d, "_iter_source_lines", iter_fn)
    ok, path = d.download_file("https://h/p/shard.gz", tmp_path)
    assert ok is True and path == tmp_path / "shard.json"
    assert called["n"] == 0  # per-shard resume: did not re-stream


def test_download_file_resigns_then_succeeds(tmp_path, monkeypatch):
    d = _bare()
    calls = []

    def iter_fn(url):
        calls.append(url)
        if "stale" in url:
            raise requests.exceptions.HTTPError("403 Forbidden", response=_Resp(403))
        yield b'{"corpusid": 1}\n'

    monkeypatch.setattr(d, "_iter_source_lines", iter_fn)
    ok, path = d.download_file(
        "https://h/p/shard.gz?sig=stale",
        tmp_path,
        resign=lambda: "https://h/p/shard.gz?sig=fresh",
    )
    assert ok is True
    assert calls == ["https://h/p/shard.gz?sig=stale", "https://h/p/shard.gz?sig=fresh"]
    assert not (tmp_path / "shard.json.partial").exists()


def test_download_file_without_resign_returns_false_on_rejection(tmp_path, monkeypatch):
    d = _bare()
    monkeypatch.setattr(d, "_iter_source_lines", _raising_iter)
    ok, path = d.download_file("https://h/p/shard.gz", tmp_path)  # no resign callback
    assert ok is False and path is None
    assert not (tmp_path / "shard.json.partial").exists()  # cleaned up, never raises


def test_download_dataset_wires_resign_to_get_dataset_info(tmp_path):
    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    d.base_dir = tmp_path
    infos = iter([
        {"files": ["https://h/p/shardA.gz?sig=stale"]},  # initial existence check
        {"files": ["https://h/p/shardA.gz?sig=fresh"]},  # re-sign refresh
    ])
    info_calls = {"n": 0}

    def fake_info(dataset, release_id):
        info_calls["n"] += 1
        return next(infos)

    d.get_dataset_info = fake_info

    seen = []

    def fake_download_file(url, output_dir, desc=None, resign=None):
        seen.append(url)
        assert resign is not None  # download_dataset must supply a re-sign callback
        fresh = resign()
        seen.append(("resigned->", fresh))
        (output_dir / "shardA.json").write_text("{}", encoding="utf-8")
        return True, output_dir / "shardA.json"

    d.download_file = fake_download_file

    ok = d.download_dataset("papers", "2026-05-26", index=False)
    assert ok is True
    assert ("resigned->", "https://h/p/shardA.gz?sig=fresh") in seen
    assert info_calls["n"] == 2  # initial check + one re-sign
