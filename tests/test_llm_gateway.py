import asyncio

import pytest

from app.config.settings import Config
from app.services.gateway_factory import GatewayFactory
from app.services.llm.gateway import ContextOverflowError, InvalidJSONResponseError, LLMGateway, LLMTask
from app.services.llm.types import ProviderResponse


class FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


class FakeProvider:
    def __init__(self, mode: str):
        self.mode = mode
        self.calls = 0
        self.last_request = None

    async def chat(self, request):
        self.calls += 1
        self.last_request = request
        if self.mode == "async_timeout":
            raise asyncio.TimeoutError("gateway timeout")
        if self.mode == "empty_length":
            # Reasoning model that burned its whole output budget on hidden
            # chain-of-thought: no content, finish_reason=length.
            return ProviderResponse(
                raw_text="",
                model_used=request.model,
                finish_reason="length",
                usage={"input_tokens": 700, "output_tokens": 800, "total_tokens": 1500, "is_estimated": False},
                reasoning="lots of thinking that never produced an answer",
            )
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
        if self.mode == "reject_json_schema":
            rf = request.response_format
            if isinstance(rf, dict) and rf.get("type") == "json_schema":
                raise FakeStatusError(400, "response_format.json_schema is not supported by this provider")
            return ProviderResponse(
                raw_text='{"ok": true}',
                model_used=request.model,
                finish_reason="stop",
                usage={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "is_estimated": False},
            )
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


def build_gateway(
    monkeypatch,
    tmp_path,
    provider,
    routing=None,
    model_registry_overrides=None,
    *,
    provider_name="openai",
    base_url="https://api.openai.com/v1",
    timeout_seconds=5,
    timeout_seconds_local=None,
):
    monkeypatch.setattr(Config, "TRACE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(Config, "TRACE_ENABLED", True, raising=False)
    monkeypatch.setattr(Config, "TRACE_EMBED_MODE", "full", raising=False)
    monkeypatch.setattr(Config, "TRACE_EMBED_MAX_BYTES", 1_000_000, raising=False)
    monkeypatch.setattr(Config, "LLM_PROVIDER", provider_name, raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(Config, "LLM_EVALUATION_MODEL", "test-model", raising=False)
    monkeypatch.setattr(Config, "LLM_BASE_URL", base_url, raising=False)
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
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS", timeout_seconds, raising=False)
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS_LOCAL", timeout_seconds_local, raising=False)
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


def test_empty_content_with_length_finish_raises_actionable_error(monkeypatch, tmp_path):
    # A model that returns no content because it hit the output-token limit must
    # surface the real cause and name the exact config knob to raise — not a
    # misleading "invalid JSON" message.
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("empty_length"))

    async def run():
        with pytest.raises(InvalidJSONResponseError) as excinfo:
            await gateway.chat_json(
                user_prompt="Return json",
                system_prompt="Return valid JSON only",
                task=LLMTask.GENERIC,
                batch_id="b7",
                claim_id="c7",
            )
        message = str(excinfo.value)
        assert "no content" in message
        assert "finish_reason=length" in message
        # Points at the model-level knob (Providers page Max Output Tokens),
        # distinguishes it from the context window, and reports the reasoning.
        assert "max_output_tokens" in message
        assert "Max Output Tokens" in message
        assert "Providers page" in message
        assert "context window" in message
        assert "characters of reasoning" in message

    asyncio.run(run())


def test_output_budget_defaults_to_model_max_output(monkeypatch, tmp_path):
    # With no per-task override, the budget is the model's Max Output Tokens
    # (Providers page) — not a hidden per-task constant.
    provider = FakeProvider("ok")
    gateway = build_gateway(
        monkeypatch, tmp_path, provider=provider,
        model_registry_overrides={
            "test-model": {
                "context_window_tokens": 256000,
                "max_output_tokens_default": 16000,
                "supports_temperature": True,
                "supports_json_schema_or_json_mode": True,
            }
        },
    )

    async def run():
        await gateway.chat_json(
            user_prompt="Return json", system_prompt="Return valid JSON only",
            task=LLMTask.QUERY_GENERATION, batch_id="b8a", claim_id="c8a",
        )
        # Previously this would have been a hardcoded 800 for query generation.
        assert provider.last_request.max_output_tokens == 16000

    asyncio.run(run())


def test_routing_task_override_still_wins_over_model_default(monkeypatch, tmp_path):
    # An explicit per-task override (advanced) still takes precedence, even with
    # routing disabled for model selection.
    provider = FakeProvider("ok")
    gateway = build_gateway(
        monkeypatch, tmp_path, provider=provider,
        routing={"enabled": False, "tasks": {LLMTask.GENERIC: {"max_output_tokens": 9000}}},
        model_registry_overrides={
            "test-model": {
                "context_window_tokens": 256000,
                "max_output_tokens_default": 512,
                "supports_temperature": True,
                "supports_json_schema_or_json_mode": True,
            }
        },
    )

    async def run():
        await gateway.chat_json(
            user_prompt="Return json", system_prompt="Return valid JSON only",
            task=LLMTask.GENERIC, batch_id="b8", claim_id="c8",
        )
        assert provider.last_request.max_output_tokens == 9000

    asyncio.run(run())


def test_timeout_override_is_ignored_when_routing_disabled(monkeypatch, tmp_path):
    routing = {
        "enabled": False,
        "tasks": {
            LLMTask.FINAL_REPORT: {
                "timeout_seconds": 300,
            }
        },
    }
    gateway = build_gateway(
        monkeypatch,
        tmp_path,
        provider=FakeProvider("success"),
        routing=routing,
        timeout_seconds=45,
    )

    timeout_value, timeout_source = gateway._resolve_timeout_details(LLMTask.FINAL_REPORT)

    assert timeout_value == 45
    assert timeout_source == "global_default"


def test_timeout_override_is_used_when_routing_enabled(monkeypatch, tmp_path):
    routing = {
        "enabled": True,
        "tasks": {
            LLMTask.FINAL_REPORT: {
                "timeout_seconds": 300,
            }
        },
    }
    gateway = build_gateway(
        monkeypatch,
        tmp_path,
        provider=FakeProvider("success"),
        routing=routing,
        timeout_seconds=45,
    )

    timeout_value, timeout_source = gateway._resolve_timeout_details(LLMTask.FINAL_REPORT)

    assert timeout_value == 300
    assert timeout_source == "task_override"


def test_loopback_openai_base_url_uses_local_timeout(monkeypatch, tmp_path):
    gateway = build_gateway(
        monkeypatch,
        tmp_path,
        provider=FakeProvider("success"),
        timeout_seconds=45,
        timeout_seconds_local=600,
        base_url="http://127.0.0.1:11434/v1",
    )

    timeout_value, timeout_source = gateway._resolve_timeout_details(LLMTask.FINAL_REPORT)

    assert timeout_value == 600
    assert timeout_source == "local_default"


def test_ollama_context_initialization_skips_missing_default_model(monkeypatch, tmp_path):
    gateway = build_gateway(
        monkeypatch,
        tmp_path,
        provider=FakeProvider("success"),
        provider_name="ollama",
        base_url="http://localhost:11434",
    )
    gateway.default_model = None

    asyncio.run(gateway._initialize_ollama_context())

    assert gateway._ollama_show_error is None


def test_gateway_factory_falls_back_when_provider_default_model_is_null(monkeypatch):
    monkeypatch.setattr(Config, "LLM_EVALUATION_MODEL", "llama3.1:8b", raising=False)

    runtime_config = GatewayFactory()._build_runtime_config({"default_model": None})

    assert runtime_config["default_model"] == "llama3.1:8b"


def test_gateway_factory_can_use_first_enabled_provider_model(monkeypatch):
    monkeypatch.setattr(Config, "LLM_EVALUATION_MODEL", "", raising=False)
    provider_snapshot = {
        "default_model": None,
        "models": [
            {"model_name": "disabled-model", "enabled": False},
            {"model_name": "gemma4:31b", "enabled": True},
        ],
    }

    runtime_config = GatewayFactory()._build_runtime_config(provider_snapshot)

    assert runtime_config["default_model"] == "gemma4:31b"


SIMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


def test_strict_schema_sends_json_schema_response_format(monkeypatch, tmp_path):
    # Strict mode ON + a schema supplied -> provider receives a json_schema
    # response_format that pins the exact field names.
    provider = FakeProvider("success")
    gateway = build_gateway(monkeypatch, tmp_path, provider=provider)
    monkeypatch.setattr(Config, "LLM_STRICT_JSON_SCHEMA", True, raising=False)

    async def run():
        await gateway.chat_json(
            user_prompt="Return json", system_prompt="Return valid JSON only",
            task=LLMTask.FINAL_REPORT, batch_id="bs1", claim_id="cs1",
            response_schema=SIMPLE_SCHEMA, schema_name="simple",
        )
        rf = provider.last_request.response_format
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "simple"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == SIMPLE_SCHEMA

    asyncio.run(run())


def test_strict_schema_without_schema_uses_json_object(monkeypatch, tmp_path):
    # Strict mode ON but no schema passed (e.g. a stage we didn't wire) -> plain
    # json_object, so "valid JSON" enforcement still applies.
    provider = FakeProvider("success")
    gateway = build_gateway(monkeypatch, tmp_path, provider=provider)
    monkeypatch.setattr(Config, "LLM_STRICT_JSON_SCHEMA", True, raising=False)

    async def run():
        await gateway.chat_json(
            user_prompt="Return json", system_prompt="Return valid JSON only",
            task=LLMTask.GENERIC, batch_id="bs2", claim_id="cs2",
        )
        assert provider.last_request.response_format == {"type": "json_object"}

    asyncio.run(run())


def test_strict_schema_disabled_uses_json_object_even_with_schema(monkeypatch, tmp_path):
    provider = FakeProvider("success")
    gateway = build_gateway(monkeypatch, tmp_path, provider=provider)
    monkeypatch.setattr(Config, "LLM_STRICT_JSON_SCHEMA", False, raising=False)

    async def run():
        await gateway.chat_json(
            user_prompt="Return json", system_prompt="Return valid JSON only",
            task=LLMTask.FINAL_REPORT, batch_id="bs3", claim_id="cs3",
            response_schema=SIMPLE_SCHEMA, schema_name="simple",
        )
        assert provider.last_request.response_format == {"type": "json_object"}

    asyncio.run(run())


def test_json_schema_rejection_downgrades_to_json_object(monkeypatch, tmp_path):
    # A provider that doesn't support json_schema must not hard-fail strict mode:
    # the gateway retries the same call with plain json_object.
    provider = FakeProvider("reject_json_schema")
    gateway = build_gateway(monkeypatch, tmp_path, provider=provider)
    monkeypatch.setattr(Config, "LLM_STRICT_JSON_SCHEMA", True, raising=False)

    async def run():
        result = await gateway.chat_json(
            user_prompt="Return json", system_prompt="Return valid JSON only",
            task=LLMTask.FINAL_REPORT, batch_id="bs4", claim_id="cs4",
            response_schema=SIMPLE_SCHEMA, schema_name="simple",
        )
        assert result["content"]["ok"] is True
        # Two provider calls: rejected json_schema, then accepted json_object.
        assert provider.calls == 2
        assert provider.last_request.response_format == {"type": "json_object"}

    asyncio.run(run())


def test_json_repair_enabled_property_reflects_config(monkeypatch, tmp_path):
    gateway = build_gateway(monkeypatch, tmp_path, provider=FakeProvider("success"))
    monkeypatch.setattr(Config, "LLM_JSON_REPAIR_PASS", False, raising=False)
    assert gateway.json_repair_enabled is False
    monkeypatch.setattr(Config, "LLM_JSON_REPAIR_PASS", True, raising=False)
    assert gateway.json_repair_enabled is True


def test_repair_pass_reformats_content_against_schema(monkeypatch, tmp_path):
    # repair_json_to_schema asks the model to reshape bad content and returns the
    # reformatted JSON, sending the schema as a json_schema response_format.
    provider = FakeProvider("success")
    gateway = build_gateway(monkeypatch, tmp_path, provider=provider)
    monkeypatch.setattr(Config, "LLM_STRICT_JSON_SCHEMA", True, raising=False)

    async def run():
        result = await gateway.repair_json_to_schema(
            bad_content={"wrong_field": "value"},
            response_schema=SIMPLE_SCHEMA,
            schema_name="simple",
            task=LLMTask.FINAL_REPORT,
            batch_id="bs5", claim_id="cs5",
            stage=LLMTask.FINAL_REPORT,
        )
        assert result["content"]["ok"] is True
        # The repair prompt carries the schema and the original data.
        sent = provider.last_request.messages[-1]["content"]
        assert "wrong_field" in sent
        assert "REQUIRED JSON SCHEMA" in sent
        assert provider.last_request.response_format["type"] == "json_schema"

    asyncio.run(run())


def test_timeout_failure_records_timeout_diagnostics(monkeypatch, tmp_path):
    gateway = build_gateway(
        monkeypatch,
        tmp_path,
        provider=FakeProvider("async_timeout"),
        timeout_seconds=9,
    )
    monkeypatch.setattr(gateway.retry_policy, "max_retries", 0)

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await gateway.chat_json(
                user_prompt="Return json",
                system_prompt="Return valid JSON only",
                task=LLMTask.GENERIC,
                batch_id="b7",
                claim_id="c7",
            )
        traces = await gateway.get_claim_traces("b7", "c7")
        assert len(traces) == 1
        assert traces[0]["status"] == "error"
        assert traces[0]["timed_out"] is True
        assert traces[0]["timeout_source"] == "global_default"
        issues = await gateway.get_claim_issues("b7", "c7")
        assert issues[0]["details"]["timed_out"] is True
        assert issues[0]["details"]["timeout_source"] == "global_default"

    asyncio.run(run())
