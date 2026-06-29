"""Tests for the on-miss remote content fetch (FETCH_REMOTE_CONTENT_ON_MISS)."""
import sys
import types

sys.modules.setdefault("ijson", types.SimpleNamespace())
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from semantic_scholar.utils.searcher import S2Searcher


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = b"x"

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


def _bare_searcher():
    s = S2Searcher.__new__(S2Searcher)
    s._last_remote_fetch = 0.0
    s._remote_min_interval = 0.0  # no real waiting in tests
    return s


def test_remote_fetch_returns_abstract(monkeypatch):
    s = _bare_searcher()
    captured = {}

    class _Session:
        def get(self, url, params=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            return _Resp(200, {"abstract": "The paper abstract.", "tldr": None})

    s.session = _Session()
    result = s._fetch_remote_content("12345")
    assert result == {"text": "The paper abstract.", "source": "semantic_scholar_api_abstract"}
    assert "CorpusId:12345" in captured["url"]


def test_remote_fetch_falls_back_to_tldr(monkeypatch):
    s = _bare_searcher()

    class _Session:
        def get(self, url, params=None, timeout=None):
            return _Resp(200, {"abstract": None, "tldr": {"text": "Short summary."}})

    s.session = _Session()
    result = s._fetch_remote_content("9")
    assert result == {"text": "Short summary.", "source": "semantic_scholar_api_tldr"}


def test_remote_fetch_no_text_reports_clean_miss():
    s = _bare_searcher()

    class _Session:
        def get(self, url, params=None, timeout=None):
            return _Resp(200, {"abstract": None, "tldr": None})

    s.session = _Session()
    assert s._fetch_remote_content("9")["status"] == "remote_no_text"


def test_remote_fetch_404_is_not_found():
    s = _bare_searcher()

    class _Session:
        def get(self, url, params=None, timeout=None):
            return _Resp(404)

    s.session = _Session()
    assert s._fetch_remote_content("9")["status"] == "remote_not_found"


def test_remote_fetch_retries_on_429(monkeypatch):
    import semantic_scholar.utils.searcher as searcher_module
    monkeypatch.setattr(searcher_module.time, "sleep", lambda *_a, **_k: None)
    s = _bare_searcher()
    calls = {"n": 0}

    class _Session:
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                return _Resp(429)
            return _Resp(200, {"abstract": "Recovered."})

    s.session = _Session()
    result = s._fetch_remote_content("9")
    assert result == {"text": "Recovered.", "source": "semantic_scholar_api_abstract"}
    assert calls["n"] == 3


def test_get_paper_content_uses_remote_only_when_enabled(monkeypatch):
    import semantic_scholar.utils.searcher as searcher_module

    s = _bare_searcher()
    s.current_release = "rel"
    s.has_local_data = True
    # All local lookups miss.
    s.indexer = types.SimpleNamespace(lookup=lambda **kwargs: None)
    remote_calls = {"n": 0}

    def fake_remote(corpus_id):
        remote_calls["n"] += 1
        return {"text": "Remote abstract.", "source": "semantic_scholar_api_abstract"}

    s._fetch_remote_content = fake_remote

    # Toggle OFF: no remote fetch, content inaccessible.
    monkeypatch.setattr(searcher_module.Config, "FETCH_REMOTE_CONTENT_ON_MISS", False, raising=False)
    result = s.get_paper_content("123")
    assert remote_calls["n"] == 0
    assert result.get("text") is None
    assert result.get("reason_code") == "no_accessible_content"

    # Toggle ON: remote fetch fills the content.
    monkeypatch.setattr(searcher_module.Config, "FETCH_REMOTE_CONTENT_ON_MISS", True, raising=False)
    result = s.get_paper_content("123")
    assert remote_calls["n"] == 1
    assert result["text"] == "Remote abstract."
    assert result["source"] == "semantic_scholar_api_abstract"


def test_fetch_remote_author_metadata_batches_ids():
    s = _bare_searcher()
    captured = {}

    class _Session:
        def post(self, url, params=None, json=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            captured["ids"] = json["ids"]
            return _Resp(200, [
                {"authorId": "1", "hIndex": 42, "paperCount": 100, "citationCount": 9000},
                None,  # API returns null for unknown authors
            ])

    s.session = _Session()
    result = s._fetch_remote_author_metadata(["1", "2"])
    assert captured["ids"] == ["1", "2"]
    assert "author/batch" in captured["url"]
    assert result["1"]["hIndex"] == 42
    assert "2" not in result  # null entry skipped


def test_fetch_remote_author_metadata_empty_ids_skips_call():
    s = _bare_searcher()

    class _Session:
        def post(self, *a, **k):
            raise AssertionError("should not call the API with no ids")

    s.session = _Session()
    assert s._fetch_remote_author_metadata([]) == {}
    assert s._fetch_remote_author_metadata([None, ""]) == {}


def test_enrich_author_data_uses_remote_only_when_enabled(monkeypatch):
    import semantic_scholar.utils.searcher as searcher_module

    s = _bare_searcher()
    s.current_release = "rel"
    # Local authors dataset misses every lookup (mini corpus doesn't have them).
    s.indexer = types.SimpleNamespace(lookup=lambda **kwargs: None)
    remote_calls = {"ids": None}

    def fake_remote(ids):
        remote_calls["ids"] = list(ids)
        return {"55": {"hIndex": 73, "paperCount": 300, "citationCount": 25000}}

    s._fetch_remote_author_metadata = fake_remote
    authors = [{"authorId": "55", "name": "F. J. van Spronsen"}]

    # Toggle OFF: no remote fetch, h-index stays unset (formatter would show 0).
    monkeypatch.setattr(searcher_module.Config, "FETCH_REMOTE_CONTENT_ON_MISS", False, raising=False)
    enriched = s._enrich_author_data([dict(a) for a in authors])
    assert remote_calls["ids"] is None
    assert "hIndex" not in enriched[0]

    # Toggle ON: remote fetch fills the real h-index.
    monkeypatch.setattr(searcher_module.Config, "FETCH_REMOTE_CONTENT_ON_MISS", True, raising=False)
    enriched = s._enrich_author_data([dict(a) for a in authors])
    assert remote_calls["ids"] == ["55"]
    assert enriched[0]["hIndex"] == 73
    assert enriched[0]["citationCount"] == 25000


def test_enrich_author_data_prefers_local_over_remote(monkeypatch):
    import semantic_scholar.utils.searcher as searcher_module

    s = _bare_searcher()
    s.current_release = "rel"
    s.indexer = types.SimpleNamespace(
        lookup=lambda **kwargs: {"hindex": 12, "papercount": 40, "citationcount": 500}
    )

    def fail_remote(ids):
        raise AssertionError("remote must not be called when local hits")

    s._fetch_remote_author_metadata = fail_remote
    monkeypatch.setattr(searcher_module.Config, "FETCH_REMOTE_CONTENT_ON_MISS", True, raising=False)
    enriched = s._enrich_author_data([{"authorId": "9", "name": "Local Author"}])
    assert enriched[0]["hIndex"] == 12
