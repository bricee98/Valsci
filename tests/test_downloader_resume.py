import gzip
import io
import json
import sys
import types
from pathlib import Path

import pytest
import urllib3


sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils import downloader as downloader_module
from semantic_scholar.utils.downloader import (
    ResumeNotSupported,
    S2DatasetDownloader,
    _ResumableHTTPReader,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Don't actually wait through resume/restart backoff in tests.
    monkeypatch.setattr(downloader_module.time, "sleep", lambda *_a, **_k: None)


class FakeRaw:
    """Serves bytes from an offset; optionally raises a transient drop once."""

    def __init__(self, data: bytes, drop_after=None):
        self.data = data
        self.i = 0
        self.drop_after = drop_after
        self.dropped = False

    def read(self, amt=-1, decode_content=False):
        if amt is None or amt < 0:
            amt = len(self.data) - self.i
        if self.drop_after is not None and not self.dropped and self.i >= self.drop_after:
            self.dropped = True
            raise urllib3.exceptions.ProtocolError("simulated connection drop")
        end = self.i + amt
        if self.drop_after is not None and not self.dropped:
            end = min(end, self.drop_after)
        chunk = self.data[self.i:end]
        self.i = end
        return chunk


class FakeResponse:
    def __init__(self, data, status_code, drop_after=None):
        self.raw = FakeRaw(data, drop_after=drop_after)
        self.status_code = status_code

    def close(self):
        pass


def _make_downloader(script):
    """Build a bare downloader whose make_request follows a scripted list of
    (status_code, drop_after) responses, serving from the given offset."""
    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    d.get_filename_from_url = lambda url: "shard.gz"
    calls = []

    def fake_make_request(url, method="get", max_retries=5, **kwargs):
        headers = kwargs.get("headers") or {}
        rng = headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        idx = len(calls)
        calls.append({"range": rng, "start": start})
        status, drop_after, payload = script(idx, start)
        return FakeResponse(payload, status, drop_after=drop_after)

    d.make_request = fake_make_request
    d._calls = calls
    return d


def _gz_payload(n=40000):
    rows = [{"corpusid": i, "name": f"author-{i}-zzz"} for i in range(n)]
    raw = ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")
    return raw, gzip.compress(raw)


def test_resumable_reader_reconstructs_exact_bytes():
    _, gz = _gz_payload()
    drop = len(gz) // 2
    assert drop > 8192, "payload must exceed one buffer so the drop is mid-stream"

    def script(idx, start):
        if idx == 0:
            return 200, drop, gz  # initial request drops mid-stream
        return 206, None, gz[start:]  # resume serves the remainder

    d = _make_downloader(script)
    reader = _ResumableHTTPReader(d, "https://example.org/shard.gz")
    got = io.BufferedReader(reader).read()
    assert got == gz
    assert len(d._calls) >= 2 and d._calls[1]["range"] is not None


def test_iter_source_lines_recovers_all_lines_after_drop():
    raw, gz = _gz_payload()
    expected = raw.decode("utf-8").splitlines()
    drop = len(gz) // 2

    def script(idx, start):
        if idx == 0:
            return 200, drop, gz
        return 206, None, gz[start:]

    d = _make_downloader(script)
    lines = [l.decode("utf-8").rstrip("\n") for l in d._iter_source_lines("https://example.org/shard.gz")]
    assert lines == expected


def test_whole_shard_restart_when_range_ignored():
    raw, gz = _gz_payload()
    expected_count = len(raw.decode("utf-8").splitlines())
    drop = len(gz) // 2

    def script(idx, start):
        if idx == 0:
            return 200, drop, gz       # initial drop mid-stream
        if idx == 1:
            return 200, None, gz[start:]  # resume answered 200 (Range ignored) -> ResumeNotSupported
        return 200, None, gz           # whole restart from the beginning succeeds

    d = _make_downloader(script)
    lines = list(d._iter_source_lines("https://example.org/shard.gz"))
    assert len(lines) == expected_count
    # initial + failed-resume + full restart
    assert len(d._calls) == 3
    assert d._calls[1]["range"] is not None  # the resume attempt used Range
    assert d._calls[2]["range"] is None      # the restart began from scratch


def test_resume_gives_up_after_max_attempts():
    _, gz = _gz_payload()
    drop = len(gz) // 2

    def script(idx, start):
        if idx == 0:
            return 200, drop, gz  # initial drops mid-stream
        return 206, 0, gz[start:]  # every resume drops immediately (no progress)

    d = _make_downloader(script)
    reader = _ResumableHTTPReader(d, "https://example.org/shard.gz", max_resume_attempts=2)
    with pytest.raises(urllib3.exceptions.ProtocolError):
        io.BufferedReader(reader).read()
    # 1 initial open + max_resume_attempts reopens, then it gives up
    assert len(d._calls) == 1 + 2
