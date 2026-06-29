import sys
import types

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils import downloader as downloader_module
from semantic_scholar.utils.downloader import S2DatasetDownloader


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass


def _bare_downloader():
    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    d.session = types.SimpleNamespace(headers={})
    d.rate_limiter = types.SimpleNamespace(wait=lambda: None)
    return d


def test_make_request_applies_default_timeout(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(downloader_module.requests, "get", fake_get)
    _bare_downloader().make_request("https://example.org/shard.gz", stream=True)
    assert captured["timeout"] == downloader_module.DEFAULT_REQUEST_TIMEOUT


def test_make_request_preserves_explicit_timeout(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(downloader_module.requests, "get", fake_get)
    _bare_downloader().make_request("https://example.org/shard.gz", timeout=(5, 5))
    assert captured["timeout"] == (5, 5)


def test_default_read_timeout_is_finite_and_in_transient_errors():
    # A read timeout only helps if it actually fires and is recoverable: the
    # resume machinery keys off urllib3 ReadTimeoutError being transient.
    import urllib3

    connect, read = downloader_module.DEFAULT_REQUEST_TIMEOUT
    assert read is not None and read > 0
    assert urllib3.exceptions.ReadTimeoutError in downloader_module.TRANSIENT_STREAM_ERRORS


def test_shard_progress_logger_throttles_by_lines_and_time(monkeypatch):
    prints = []
    monkeypatch.setattr(downloader_module.console, "print", lambda msg: prints.append(msg))
    clock = {"t": 1000.0}
    monkeypatch.setattr(downloader_module.time, "time", lambda: clock["t"])

    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    tick = d._shard_progress_logger(
        "papers 13/60",
        lambda: "5 matched, 2 still missing",
        every_lines=10,
        every_seconds=15.0,
    )

    # Below the line gate: silent.
    for _ in range(9):
        tick()
    assert prints == []

    # Hits the line gate at row 10, but no time has elapsed yet: still silent.
    tick()
    assert prints == []

    # Enough wall-clock has passed; the next line-gate hit (row 20) prints once.
    clock["t"] += 20
    for _ in range(10):
        tick()
    assert len(prints) == 1
    message = prints[0]
    assert "papers 13/60" in message
    assert "20 rows scanned" in message
    assert "5 matched, 2 still missing" in message


def test_shard_progress_logger_quiet_for_small_shards(monkeypatch):
    prints = []
    monkeypatch.setattr(downloader_module.console, "print", lambda msg: prints.append(msg))
    monkeypatch.setattr(downloader_module.time, "time", lambda: 0.0)

    d = S2DatasetDownloader.__new__(S2DatasetDownloader)
    tick = d._shard_progress_logger("authors 1/2", every_lines=100_000)
    for _ in range(5_000):
        tick()
    assert prints == []
