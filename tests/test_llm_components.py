import asyncio
import json

from app.services.llm.rate_limiter import GatewayRateLimiter
from app.services.llm.retry_policy import RetryPolicy
from app.services.llm.token_estimator import TokenEstimator
from app.services.llm.trace_store import TraceStore


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
