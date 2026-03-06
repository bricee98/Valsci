import asyncio

import pytest

from app.config.settings import Config
from app.services.llm.gateway import ContextOverflowError, LLMGateway, LLMTask
from app.services.llm.types import ProviderResponse


class FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


class FakeProvider:
    def __init__(self, mode: str):
        self.mode = mode
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        if self.mode == "retry_then_success":
            if self.calls <= 2:
                raise FakeStatusError(429, "rate limited")
            return ProviderResponse(
                raw_text='{"ok": true}',
                model_used=request.model,
                finish_reason="stop",
                usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "is_estimated": False},
            )
        if self.mode == "invalid_json_once":
            if self.calls == 1:
                return ProviderResponse(
                    raw_text='{"ok":',
                    model_used=request.model,
                    finish_reason="stop",
                    usage={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60, "is_estimated": False},
                )
            return ProviderResponse(
                raw_text='{"ok": "fixed"}',
                model_used=request.model,
                finish_reason="stop",
                usage={"input_tokens": 60, "output_tokens": 15, "total_tokens": 75, "is_estimated": False},
            )
        if self.mode == "bad_request":
            raise FakeStatusError(400, "bad request")
        if self.mode == "fenced_json":
            return ProviderResponse(
                raw_text='```json\n{"ok": true}\n```',
                model_used=request.model,
                finish_reason="stop",
                usage={"input_tokens": 30, "output_tokens": 8, "total_tokens": 38, "is_estimated": False},
            )
        return ProviderResponse(
            raw_text='{"ok": true}',
            model_used=request.model,
            finish_reason="stop",
            usage={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "is_estimated": False},
        )


def build_gateway(monkeypatch, tmp_path, provider, routing=None, model_registry_overrides=None):
    monkeypatch.setattr(Config, "TRACE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(Config, "TRACE_ENABLED", True, raising=False)
    monkeypatch.setattr(Config, "TRACE_EMBED_MODE", "full", raising=False)
    monkeypatch.setattr(Config, "TRACE_EMBED_MAX_BYTES", 1_000_000, raising=False)
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(Config, "LLM_EVALUATION_MODEL", "test-model", raising=False)
    monkeypatch.setattr(Config, "LLM_ROUTING", routing or {"enabled": False}, raising=False)
    monkeypatch.setattr(
        Config,
        "MODEL_REGISTRY_OVERRIDES",
        model_registry_overrides
        or {
            "test-model": {
                "context_window_tokens": 4096,
                "max_output_tokens_default": 512,
                "supports_temperature": True,
                "supports_json_schema_or_json_mode": True,
            }
        },
        raising=False,
    )
    monkeypatch.setattr(Config, "LLM_MAX_CONCURRENCY", 2, raising=False)
    monkeypatch.setattr(Config, "LLM_REQUESTS_PER_MINUTE", 1000, raising=False)
    monkeypatch.setattr(Config, "LLM_TOKENS_PER_MINUTE", 1_000_000, raising=False)
    monkeypatch.setattr(Config, "LLM_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_BASE_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_MAX_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(Config, "LLM_BACKOFF_JITTER", 0.0, raising=False)
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS", 5, raising=False)
    monkeypatch.setattr(LLMGateway, "_build_provider", lambda self: provider)
    return LLMGateway()


def test_context_preflight_overflow_creates_issue(monkeypatch, tmp_path):
    gateway = build_gateway(
        monkeypatch,
        tmp_path,
        provider=FakeProvider("success"),
        model_registry_overrides={
            "test-model": {
                "context_window_tokens": 300,
                "max_output_tokens_default": 100,
                "supports_temperature": True,
                "supports_json_schema_or_json_mode": True,
            }
        },
    )

    async def run():
        with pytest.raises(ContextOverflowError):
            await gateway._route_and_preflight(
                task=LLMTask.QUERY_GENERATION,
                messages=[{"role": "user", "content": "x" * 5000}],
                model_override=None,
                max_output_tokens=200,
                locked_models=False,
                batch_id="b1",
                claim_id="c1",
                paper_id=None,
                stage="query_generation",
            )
        issues = await gateway.get_claim_issues("b1", "c1")
        assert any("Context overflow prevented" in issue["message"] for issue in issues)

    asyncio.run(run())


def test_router_uses_fallback_when_preferred_cannot_fit(monkeypatch, tmp_path):
    routing = {
        "enabled": True,
        "tasks": {
            "query_generation": {
                "preferred_models": ["small-model"],
                "fallback_models": ["large-model"],
                "max_output_tokens": 200,
            }
        },
    }
    overrides = {
        "small-model": {
            "context_window_tokens": 500,
            "max_output_tokens_default": 200,
            "supports_temperature": True,
            "supports_json_schema_or_json_mode": True,
        },
        "large-model": {
            "context_window_tokens": 8000,
            "max_output_tokens_default": 200,
            "supports_temperature": True,
            "supports_json_schema_or_json_mode": True,
        },
        "test-model": {
            "context_window_tokens": 8000,
            "max_output_tokens_default": 200,
            "supports_temperature": True,
            "supports_json_schema_or_json_mode": True,
        },
    }
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("success"), routing=routing, model_registry_overrides=overrides)

    async def run():
        result = await gateway._route_and_preflight(
            task=LLMTask.QUERY_GENERATION,
            messages=[{"role": "user", "content": "x" * 2500}],
            model_override=None,
            max_output_tokens=200,
            locked_models=False,
            batch_id="b2",
            claim_id="c2",
            paper_id=None,
            stage="query_generation",
        )
        assert result.model == "large-model"
        assert "fallback" in result.reason

    asyncio.run(run())


def test_retry_traces_each_attempt(monkeypatch, tmp_path):
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("retry_then_success"))

    async def run():
        result = await gateway.chat_json(
            user_prompt="Return json",
            system_prompt="Return {\"ok\": true}",
            task=LLMTask.GENERIC,
            batch_id="b3",
            claim_id="c3",
        )
        assert result["content"]["ok"] is True
        traces = await gateway.get_claim_traces("b3", "c3")
        assert len(traces) == 3
        assert traces[0]["status"] == "retrying"
        assert traces[1]["status"] == "retrying"
        assert traces[2]["status"] == "success"

    asyncio.run(run())


def test_invalid_json_recovery_retry(monkeypatch, tmp_path):
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("invalid_json_once"))

    async def run():
        result = await gateway.chat_json(
            user_prompt="Return json",
            system_prompt="Return valid JSON only",
            task=LLMTask.GENERIC,
            batch_id="b4",
            claim_id="c4",
        )
        assert result["content"]["ok"] == "fixed"
        assert result["usage"]["input_tokens"] == 110
        assert result["usage"]["output_tokens"] == 25
        assert result["usage"]["total_tokens"] == 135
        traces = await gateway.get_claim_traces("b4", "c4")
        assert len(traces) == 2
        assert traces[0]["status"] == "retrying"
        assert traces[1]["status"] == "success"
        issues = await gateway.get_claim_issues("b4", "c4")
        assert any("Invalid JSON returned" in issue["message"] for issue in issues)

    asyncio.run(run())


def test_bad_request_is_not_retried(monkeypatch, tmp_path):
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("bad_request"))

    async def run():
        with pytest.raises(FakeStatusError):
            await gateway.chat_json(
                user_prompt="Return json",
                system_prompt="Return valid JSON only",
                task=LLMTask.GENERIC,
                batch_id="b5",
                claim_id="c5",
            )
        traces = await gateway.get_claim_traces("b5", "c5")
        assert len(traces) == 1
        assert traces[0]["status"] == "error"

    asyncio.run(run())


def test_fenced_json_is_recovered_without_retry(monkeypatch, tmp_path):
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("fenced_json"))

    async def run():
        result = await gateway.chat_json(
            user_prompt="Return json",
            system_prompt="Return valid JSON only",
            task=LLMTask.GENERIC,
            batch_id="b6",
            claim_id="c6",
        )
        assert result["content"]["ok"] is True
        traces = await gateway.get_claim_traces("b6", "c6")
        assert len(traces) == 1
        assert traces[0]["status"] == "success"
        issues = await gateway.get_claim_issues("b6", "c6")
        assert any("Recovered JSON object" in issue["message"] for issue in issues)

    asyncio.run(run())
