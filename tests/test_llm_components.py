import asyncio
import json
import sys
import types
import pytest

from app.config.settings import Config
from app.services.llm.rate_limiter import GatewayRateLimiter
from app.services.llm.retry_policy import RetryPolicy
from app.services.llm.token_estimator import TokenEstimator
from app.services.llm.trace_store import TraceStore


def _load_searcher_class():
    sys.modules.setdefault("ijson", types.ModuleType("ijson"))
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules.setdefault("openai", openai_stub)
    from semantic_scholar.utils.searcher import S2Searcher

    return S2Searcher


def test_token_estimator_produces_non_zero_counts():
    estimator = TokenEstimator()
    text_tokens = estimator.estimate_text_tokens("A claim about scientific evidence.")
    chat_tokens = estimator.estimate_chat_tokens(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize this claim."},
        ],
        model_hint="gpt-4o",
    )
    assert text_tokens > 0
    assert chat_tokens > text_tokens


def test_retry_policy_backoff_increases_and_is_bounded():
    policy = RetryPolicy(max_retries=5, backoff_base_seconds=1.0, backoff_max_seconds=4.0, backoff_jitter=0.0)
    delays = [policy.compute_backoff_seconds(i) for i in range(1, 6)]
    assert delays == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_trace_store_concurrent_appends_are_valid_jsonl(tmp_path):
    store = TraceStore(root_dir=str(tmp_path), enabled=True)

    async def writer(idx: int):
        await store.append("batch1", "claim1", {"trace_id": str(idx), "idx": idx})

    async def run():
        await asyncio.gather(*(writer(i) for i in range(100)))

    asyncio.run(run())
    trace_file = tmp_path / "batch1" / "traces" / "claim1.jsonl"
    assert trace_file.exists()
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    decoded = [json.loads(line) for line in lines]
    assert sorted(item["idx"] for item in decoded) == list(range(100))


def test_rate_limiter_basic_reservation():
    limiter = GatewayRateLimiter(max_concurrency=1, requests_per_minute=1000, tokens_per_minute=100000)

    async def run():
        async with limiter.reserve(estimated_tokens=10):
            return True

    assert asyncio.run(run()) is True


def test_validate_config_accepts_out_of_range_numeric_values(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "test", raising=False)
    monkeypatch.setattr(Config, "SECRET_KEY", "secret", raising=False)
    monkeypatch.setattr(Config, "USER_EMAIL", "user@example.com", raising=False)
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_API_KEY", "test", raising=False)
    monkeypatch.setattr(Config, "REQUIRE_PASSWORD", False, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_BASE_SECONDS", 10.0, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_MAX_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_JITTER", 9.0, raising=False)
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS", 999999, raising=False)

    Config.validate_config()


def test_validate_config_rejects_non_numeric_backoff(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "test", raising=False)
    monkeypatch.setattr(Config, "SECRET_KEY", "secret", raising=False)
    monkeypatch.setattr(Config, "USER_EMAIL", "user@example.com", raising=False)
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_API_KEY", "test", raising=False)
    monkeypatch.setattr(Config, "REQUIRE_PASSWORD", False, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_BASE_SECONDS", "abc", raising=False)

    with pytest.raises(ValueError):
        Config.validate_config()


def test_validate_config_allows_mock_mode_without_semantic_scholar_key(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "test", raising=False)
    monkeypatch.setattr(Config, "SECRET_KEY", "secret", raising=False)
    monkeypatch.setattr(Config, "USER_EMAIL", "user@example.com", raising=False)
    monkeypatch.setattr(Config, "SEMANTIC_SCHOLAR_API_KEY", "", raising=False)
    monkeypatch.setattr(Config, "MOCK_SEMANTIC_SCHOLAR_MODE", True, raising=False)
    monkeypatch.setattr(Config, "REQUIRE_PASSWORD", False, raising=False)

    Config.validate_config()


def test_get_paper_content_reports_missing_release():
    S2Searcher = _load_searcher_class()
    searcher = S2Searcher.__new__(S2Searcher)
    searcher.current_release = None
    searcher.has_local_data = False

    result = searcher.get_paper_content("123")

    assert result["status"] == "inaccessible"
    assert result["reason_code"] == "missing_release"
    assert result["lookup_details"]["attempts"] == []


def test_get_paper_content_reports_dataset_attempts_when_no_text():
    S2Searcher = _load_searcher_class()

    class FakeIndexer:
        def lookup(self, release_id, dataset, id_type, search_id):
            if dataset == "s2orc":
                return {"pdf_parse": {}}
            if dataset == "abstracts":
                return {"abstract": ""}
            if dataset == "tldrs":
                return {"text": ""}
            return None

    searcher = S2Searcher.__new__(S2Searcher)
    searcher.current_release = "2025-01-01"
    searcher.has_local_data = True
    searcher.indexer = FakeIndexer()

    result = searcher.get_paper_content("456")

    assert result["status"] == "inaccessible"
    assert result["reason_code"] == "no_accessible_content"
    attempts = result["lookup_details"]["attempts"]
    assert attempts == [
        {"dataset": "s2orc", "status": "record_without_text", "detail": "pdf_parse.body_text missing"},
        {"dataset": "abstracts", "status": "record_without_text", "detail": "abstract missing"},
        {"dataset": "tldrs", "status": "record_without_text", "detail": "text missing"},
    ]
