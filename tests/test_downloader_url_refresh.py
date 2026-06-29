import json
import sys
import types

import pytest
import requests

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils import downloader as downloader_module
from semantic_scholar.utils.downloader import S2DatasetDownloader, SIGNED_URL_REFRESH_SECONDS


class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error", response=self)


def _bare():
    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    d.session = types.SimpleNamespace(headers={})
    d.rate_limiter = types.SimpleNamespace(wait=lambda: None)
    return d


# --- _write_filtered_dataset skips missing (non-portable) sources -----------

def test_write_filtered_dataset_skips_missing_local_source(tmp_path):
    d = _bare()
    present = tmp_path / "papers.jsonl"
    present.write_text(json.dumps({"corpusid": 101}) + "\n", encoding="utf-8")
    # A manifest source_files path that is absolute to another machine.
    missing = "/somewhere/else/D:\\Git\\Valsci\\papers.jsonl"

    out = tmp_path / "mini.json"
    written, seen = d._write_filtered_dataset(
        dataset="papers",
        sources=[missing, str(present)],
        output_path=out,
        target_ids={"101"},
    )
    assert written == 1 and seen == {"101"}
    assert json.loads(out.read_text().strip())["corpusid"] == 101


# --- _is_signed_url_rejection ------------------------------------------------

def test_signed_url_rejection_detects_400_and_403_by_status():
    for status in (400, 403):
        exc = requests.exceptions.HTTPError("boom", response=_Resp(status))
        assert S2DatasetDownloader._is_signed_url_rejection(exc) is True


def test_signed_url_rejection_ignores_404_and_plain_errors():
    assert S2DatasetDownloader._is_signed_url_rejection(
        requests.exceptions.HTTPError("404 Not Found", response=_Resp(404))
    ) is False
    assert S2DatasetDownloader._is_signed_url_rejection(ValueError("nope")) is False


def test_signed_url_rejection_matches_status_in_message_without_response():
    # 403 path may raise without an attached response object.
    assert S2DatasetDownloader._is_signed_url_rejection(
        requests.exceptions.HTTPError("403 Forbidden: token expired")
    ) is True


# --- make_request 4xx semantics ---------------------------------------------

def test_make_request_does_not_retry_4xx_and_surfaces_body(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _Resp(400, text="<Error><Code>ExpiredToken</Code></Error>")

    monkeypatch.setattr(downloader_module.requests, "get", fake_get)
    with pytest.raises(requests.exceptions.HTTPError) as excinfo:
        _bare().make_request("https://s3/shard.gz", max_retries=5)
    assert calls["n"] == 1  # client error is deterministic: no retries
    assert "ExpiredToken" in str(excinfo.value)


def test_make_request_still_retries_5xx(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(downloader_module.time, "sleep", lambda *_a, **_k: None)

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _Resp(503, text="slow down")

    monkeypatch.setattr(downloader_module.requests, "get", fake_get)
    with pytest.raises(requests.exceptions.HTTPError):
        _bare().make_request("https://s3/shard.gz", max_retries=3)
    assert calls["n"] == 3  # server errors are retried up to max_retries


def test_make_request_retries_transient_tls_errors(monkeypatch):
    # A load-balanced API node briefly serving a bad cert is transient; the
    # request must be retried (over a generous default budget), not aborted.
    calls = {"n": 0}
    monkeypatch.setattr(downloader_module.time, "sleep", lambda *_a, **_k: None)

    def fake_get(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.SSLError("certificate has expired (_ssl.c:1016)")

    monkeypatch.setattr(downloader_module.requests, "get", fake_get)
    with pytest.raises(requests.exceptions.SSLError):
        _bare().make_request("https://api.semanticscholar.org/x")
    assert calls["n"] == 8  # the widened default retry budget rides out a blip


# --- _scan_remote_sources_for_ids URL refresh -------------------------------

def _scan_downloader(monkeypatch, info_sequence, iter_fn):
    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    seq = iter(info_sequence)
    info_calls = {"n": 0}

    def fake_info(dataset, release_id):
        info_calls["n"] += 1
        return next(seq)

    d.get_dataset_info = fake_info
    d._iter_source_lines = iter_fn
    d._info_calls = info_calls
    return d


def test_scan_reactively_resigns_on_rejected_shard(tmp_path, monkeypatch):
    # Shard A's first (stale) URL is rejected; after a re-sign it succeeds.
    info_sequence = [
        {"files": ["https://h/p/shardA.gz?sig=stale", "https://h/p/shardB.gz?sig=stale"]},
        {"files": ["https://h/p/shardA.gz?sig=fresh", "https://h/p/shardB.gz?sig=fresh"]},
    ]

    def iter_fn(source):
        if "shardA.gz" in source and "sig=stale" in source:
            raise requests.exceptions.HTTPError("400 Client Error", response=_Resp(400))
        if "shardA.gz" in source:
            yield json.dumps({"corpusid": 101}).encode()
        elif "shardB.gz" in source:
            yield json.dumps({"corpusid": 202}).encode()

    d = _scan_downloader(monkeypatch, info_sequence, iter_fn)
    out = tmp_path / "papers.jsonl"
    source, matched = d._scan_remote_sources_for_ids(
        dataset="papers", release_id="2026-05-26", target_ids={"101", "202"}, output_path=out
    )
    assert matched == {"101", "202"}
    assert d._info_calls["n"] == 2  # initial + one re-sign
    written = [json.loads(l) for l in out.read_text().splitlines()]
    assert {str(r["corpusid"]) for r in written} == {"101", "202"}


def test_scan_proactively_resigns_before_token_expiry(tmp_path, monkeypatch):
    info_sequence = [
        {"files": ["https://h/p/shardA.gz?sig=v1"]},
        {"files": ["https://h/p/shardA.gz?sig=v2"]},
    ]

    def iter_fn(source):
        yield json.dumps({"corpusid": 101}).encode()

    # First monotonic() call (signed_at) is t=0; every later call is far past the
    # refresh threshold, so a proactive re-sign fires before the shard is fetched.
    state = {"n": 0}

    def fake_monotonic():
        state["n"] += 1
        return 0.0 if state["n"] == 1 else SIGNED_URL_REFRESH_SECONDS + 100

    monkeypatch.setattr(downloader_module.time, "monotonic", fake_monotonic)

    d = _scan_downloader(monkeypatch, info_sequence, iter_fn)
    out = tmp_path / "papers.jsonl"
    _source, matched = d._scan_remote_sources_for_ids(
        dataset="papers", release_id="2026-05-26", target_ids={"101"}, output_path=out
    )
    assert matched == {"101"}
    assert d._info_calls["n"] == 2  # initial + proactive re-sign, no error needed


def test_scan_writes_checkpoint_and_resumes_skipping_done_shards(tmp_path, monkeypatch):
    info = {"files": ["https://h/p/shardA.gz?s=1", "https://h/p/shardB.gz?s=1"]}

    scanned = []

    def iter_fn(source):
        scanned.append("shardA" if "shardA" in source else "shardB")
        if "shardA" in source:
            yield json.dumps({"corpusid": 101}).encode()
        else:
            yield json.dumps({"corpusid": 202}).encode()

    d = _scan_downloader(monkeypatch, [info, info], iter_fn)
    out = tmp_path / "papers.jsonl"
    fp = S2DatasetDownloader._scan_fingerprint({"101", "202"})

    # First run scans both shards and records a checkpoint per shard.
    d._scan_remote_sources_for_ids(
        dataset="papers", release_id="R", target_ids={"101", "202"},
        output_path=out, scan_fingerprint=fp,
    )
    assert scanned == ["shardA", "shardB"]
    # On success the checkpoint is cleared (nothing left to resume).
    assert not (tmp_path / "papers.jsonl.progress.json").exists()


def test_scan_resumes_from_seeded_checkpoint(tmp_path, monkeypatch):
    info = {"files": ["https://h/p/shardA.gz?s=1", "https://h/p/shardB.gz?s=1"]}

    scanned = []

    def iter_fn(source):
        scanned.append("shardA" if "shardA" in source else "shardB")
        if "shardB" in source:
            yield json.dumps({"corpusid": 202}).encode()

    d = _scan_downloader(monkeypatch, [info, info], iter_fn)
    out = tmp_path / "papers.jsonl"
    out.write_text(json.dumps({"corpusid": 101}) + "\n", encoding="utf-8")  # shardA already cached
    fp = S2DatasetDownloader._scan_fingerprint({"101", "202"})
    # Seed: shardA already fully scanned.
    (tmp_path / "papers.jsonl.progress.json").write_text(
        json.dumps({"release_id": "R", "fingerprint": fp, "completed_shards": ["shardA.gz"]}),
        encoding="utf-8",
    )

    d._scan_remote_sources_for_ids(
        dataset="papers", release_id="R", target_ids={"202"},
        output_path=out, scan_fingerprint=fp,
    )
    assert scanned == ["shardB"]  # shardA skipped via checkpoint


def test_checkpoint_ignored_when_fingerprint_differs(tmp_path, monkeypatch):
    info = {"files": ["https://h/p/shardA.gz?s=1"]}

    scanned = []

    def iter_fn(source):
        scanned.append(source)
        yield json.dumps({"corpusid": 101}).encode()

    d = _scan_downloader(monkeypatch, [info, info], iter_fn)
    out = tmp_path / "papers.jsonl"
    out.write_text("", encoding="utf-8")
    # A checkpoint from a different target set must not cause shards to be skipped.
    (tmp_path / "papers.jsonl.progress.json").write_text(
        json.dumps({"release_id": "R", "fingerprint": "STALE", "completed_shards": ["shardA.gz"]}),
        encoding="utf-8",
    )
    d._scan_remote_sources_for_ids(
        dataset="papers", release_id="R", target_ids={"101"},
        output_path=out, scan_fingerprint=S2DatasetDownloader._scan_fingerprint({"101"}),
    )
    assert len(scanned) == 1  # shard was re-scanned despite the stale checkpoint


def test_checkpoint_persisted_when_a_later_shard_fails(tmp_path, monkeypatch):
    info = {"files": ["https://h/p/shardA.gz?s=1", "https://h/p/shardB.gz?s=1"]}

    def iter_fn(source):
        if "shardB" in source:
            raise requests.exceptions.HTTPError("500 Server Error", response=_Resp(500))
        yield json.dumps({"corpusid": 101}).encode()

    # Only one info response: a 500 is not a signed-url rejection, so no re-sign.
    d = _scan_downloader(monkeypatch, [info], iter_fn)
    out = tmp_path / "papers.jsonl"
    fp = S2DatasetDownloader._scan_fingerprint({"101", "202"})
    with pytest.raises(requests.exceptions.HTTPError):
        d._scan_remote_sources_for_ids(
            dataset="papers", release_id="R", target_ids={"101", "202"},
            output_path=out, scan_fingerprint=fp,
        )
    # shardA completed before shardB failed: its checkpoint must survive for resume.
    saved = json.loads((tmp_path / "papers.jsonl.progress.json").read_text())
    assert saved["completed_shards"] == ["shardA.gz"]
    assert saved["fingerprint"] == fp


def test_scan_raises_when_resign_does_not_help(tmp_path, monkeypatch):
    # If even a fresh URL keeps failing, the error must surface (not loop).
    info_sequence = [
        {"files": ["https://h/p/shardA.gz?sig=a"]},
        {"files": ["https://h/p/shardA.gz?sig=b"]},
    ]

    def iter_fn(source):
        raise requests.exceptions.HTTPError("400 Client Error", response=_Resp(400))
        yield  # pragma: no cover - makes this a generator

    d = _scan_downloader(monkeypatch, info_sequence, iter_fn)
    out = tmp_path / "papers.jsonl"
    with pytest.raises(requests.exceptions.HTTPError):
        d._scan_remote_sources_for_ids(
            dataset="papers", release_id="2026-05-26", target_ids={"101"}, output_path=out
        )
    assert d._info_calls["n"] == 2  # one re-sign attempt, then give up
