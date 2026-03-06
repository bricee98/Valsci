"""Central LLM gateway for routing, tracing, retrying, and reporting."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.config.settings import Config
from app.services.llm.issue_store import IssueStore
from app.services.llm.model_registry import ModelRegistry
from app.services.llm.providers import (
    AzureInferenceProvider,
    AzureOpenAIProvider,
    LlamaCppProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from app.services.llm.rate_limiter import GatewayRateLimiter
from app.services.llm.retry_policy import RetryPolicy
from app.services.llm.token_estimator import TokenEstimator
from app.services.llm.trace_store import TraceStore
from app.services.llm.types import ProviderRequest, TraceRecord, empty_usage, merge_usage, normalize_usage, utc_now_iso
from app.services.prompt_store import load_prompt

logger = logging.getLogger(__name__)


class LLMTask:
    QUERY_GENERATION = "query_generation"
    PAPER_ANALYSIS = "paper_analysis"
    VENUE_SCORING = "venue_scoring"
    FINAL_REPORT = "final_report"
    GENERIC = "generic"


DEFAULT_TASK_CONFIG = {
    LLMTask.QUERY_GENERATION: {
        "output_type": "json",
        "max_output_tokens": 800,
        "preferred_models": [],
        "fallback_models": [],
        "min_context_tokens": 4000,
    },
    LLMTask.PAPER_ANALYSIS: {
        "output_type": "json",
        "max_output_tokens": 1200,
        "preferred_models": [],
        "fallback_models": [],
        "min_context_tokens": 16000,
    },
    LLMTask.VENUE_SCORING: {
        "output_type": "json",
        "max_output_tokens": 120,
        "preferred_models": [],
        "fallback_models": [],
        "min_context_tokens": 4000,
    },
    LLMTask.FINAL_REPORT: {
        "output_type": "json",
        "max_output_tokens": 1800,
        "preferred_models": [],
        "fallback_models": [],
        "min_context_tokens": 16000,
    },
    LLMTask.GENERIC: {
        "output_type": "json",
        "max_output_tokens": 1024,
        "preferred_models": [],
        "fallback_models": [],
        "min_context_tokens": 4096,
    },
}


class ContextOverflowError(RuntimeError):
    """Raised when preflight detects no model can fit the prompt."""


class InvalidJSONResponseError(RuntimeError):
    """Raised when a JSON response cannot be parsed after recovery retry."""


@dataclass
class PreflightResult:
    model: str
    estimated_input_tokens: int
    estimated_total_tokens: int
    reserved_output_tokens: int
    context_window_tokens: int
    reason: str


class LLMGateway:
    def __init__(self):
        self.provider_name = (Config.LLM_PROVIDER or "openai").lower()
        self.local_backend = (getattr(Config, "LOCAL_BACKEND", "") or "").lower()
        self.default_model = Config.LLM_EVALUATION_MODEL
        self.base_url = getattr(Config, "LLM_BASE_URL", None)

        self.trace_store = TraceStore(root_dir=Config.TRACE_DIR, enabled=Config.TRACE_ENABLED)
        self.issue_store = IssueStore(root_dir=Config.TRACE_DIR, enabled=Config.TRACE_ENABLED)
        self.token_estimator = TokenEstimator()
        self.model_registry = ModelRegistry(
            model_overrides=getattr(Config, "MODEL_REGISTRY_OVERRIDES", {}),
            local_context_override=getattr(Config, "LOCAL_MODEL_CONTEXT_OVERRIDE", None),
        )
        self.rate_limiter = GatewayRateLimiter(
            max_concurrency=Config.LLM_MAX_CONCURRENCY,
            requests_per_minute=Config.LLM_REQUESTS_PER_MINUTE,
            tokens_per_minute=Config.LLM_TOKENS_PER_MINUTE,
        )
        self.retry_policy = RetryPolicy(
            max_retries=Config.LLM_MAX_RETRIES,
            backoff_base_seconds=Config.LLM_BACKOFF_BASE_SECONDS,
            backoff_max_seconds=Config.LLM_BACKOFF_MAX_SECONDS,
            backoff_jitter=Config.LLM_BACKOFF_JITTER,
        )

        self.timeout_seconds = Config.LLM_TIMEOUT_SECONDS
        self.timeout_seconds_local = getattr(Config, "LLM_TIMEOUT_SECONDS_LOCAL", None)
        self.routing_config = getattr(Config, "LLM_ROUTING", {}) or {}
        self.routing_enabled = bool(self.routing_config.get("enabled", False))
        self.routing_locked_models = bool(self.routing_config.get("locked_models", False))
        self.context_safety_margin_tokens = int(getattr(Config, "LLM_CONTEXT_SAFETY_MARGIN_TOKENS", 256))
        self.trace_embed_mode = getattr(Config, "TRACE_EMBED_MODE", "capped")
        self.trace_embed_max_bytes = int(getattr(Config, "TRACE_EMBED_MAX_BYTES", 2_000_000))
        self.stacktrace_max_bytes = int(getattr(Config, "TRACE_STACKTRACE_MAX_BYTES", 4000))

        self.provider = self._build_provider()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._ollama_show_error: Optional[str] = None
        self._ollama_warning_claims: set[Tuple[str, str]] = set()

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            if self._is_ollama_backend():
                await self._initialize_ollama_context()
            self._initialized = True

    async def chat_json(
        self,
        *,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        task: str = LLMTask.GENERIC,
        batch_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        stage: Optional[str] = None,
        model_override: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        locked_models: Optional[bool] = None,
        temperature: Optional[float] = 0.0,
    ) -> Dict[str, Any]:
        return await self._chat(
            expects_json=True,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            task=task,
            batch_id=batch_id,
            claim_id=claim_id,
            paper_id=paper_id,
            stage=stage or task,
            model_override=model_override,
            max_output_tokens=max_output_tokens,
            locked_models=locked_models,
            temperature=temperature,
        )

    async def chat_text(
        self,
        *,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        task: str = LLMTask.GENERIC,
        batch_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        stage: Optional[str] = None,
        model_override: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        locked_models: Optional[bool] = None,
        temperature: Optional[float] = 0.0,
    ) -> Dict[str, Any]:
        return await self._chat(
            expects_json=False,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            task=task,
            batch_id=batch_id,
            claim_id=claim_id,
            paper_id=paper_id,
            stage=stage or task,
            model_override=model_override,
            max_output_tokens=max_output_tokens,
            locked_models=locked_models,
            temperature=temperature,
        )

    async def generate_json_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.chat_json(
            user_prompt=prompt,
            system_prompt=system_prompt,
            task=kwargs.get("task", LLMTask.GENERIC),
            batch_id=kwargs.get("batch_id"),
            claim_id=kwargs.get("claim_id"),
            paper_id=kwargs.get("paper_id"),
            stage=kwargs.get("stage"),
            model_override=model or kwargs.get("model_override"),
            max_output_tokens=kwargs.get("max_output_tokens"),
            locked_models=kwargs.get("locked_models"),
            temperature=kwargs.get("temperature", 0.0),
        )

    async def generate_text_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.chat_text(
            user_prompt=prompt,
            system_prompt=system_prompt,
            task=kwargs.get("task", LLMTask.GENERIC),
            batch_id=kwargs.get("batch_id"),
            claim_id=kwargs.get("claim_id"),
            paper_id=kwargs.get("paper_id"),
            stage=kwargs.get("stage"),
            model_override=model or kwargs.get("model_override"),
            max_output_tokens=kwargs.get("max_output_tokens"),
            locked_models=kwargs.get("locked_models"),
            temperature=kwargs.get("temperature", 0.0),
        )

    async def add_issue(
        self,
        *,
        batch_id: Optional[str],
        claim_id: Optional[str],
        severity: str,
        stage: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self.issue_store.add_issue(
            batch_id=batch_id,
            claim_id=claim_id,
            severity=severity,
            stage=stage,
            message=message,
            details=details,
        )

    async def get_claim_issues(self, batch_id: str, claim_id: str) -> List[Dict[str, Any]]:
        return await self.issue_store.read_all(batch_id, claim_id)

    async def get_claim_traces(self, batch_id: str, claim_id: str) -> List[Dict[str, Any]]:
        return await self.trace_store.read_all(batch_id, claim_id)

    async def build_debug_trace(self, batch_id: str, claim_id: str) -> Dict[str, Any]:
        calls = await self.get_claim_traces(batch_id, claim_id)
        issues = await self.get_claim_issues(batch_id, claim_id)
        summary = self._build_trace_summary(calls, issues)

        embedded_calls = []
        capped = False
        mode = (self.trace_embed_mode or "capped").lower()
        if mode == "full":
            embedded_calls = calls
        elif mode == "summary":
            embedded_calls = []
        else:
            embedded_calls, capped = self._cap_trace_calls(calls, self.trace_embed_max_bytes)

        if capped:
            await self._maybe_write_trace_capped_issue(batch_id, claim_id)

        return {
            "summary": summary,
            "calls": embedded_calls,
            "trace_file": f"traces/{claim_id}.jsonl",
            "issues_file": f"issues/{claim_id}.jsonl",
        }

    def get_trace_file_path(self, batch_id: str, claim_id: str) -> str:
        return str(self.trace_store.trace_file_path(batch_id, claim_id))

    def get_issues_file_path(self, batch_id: str, claim_id: str) -> str:
        return str(self.issue_store.issue_file_path(batch_id, claim_id))

    async def _chat(
        self,
        *,
        expects_json: bool,
        user_prompt: str,
        system_prompt: Optional[str],
        task: str,
        batch_id: Optional[str],
        claim_id: Optional[str],
        paper_id: Optional[str],
        stage: str,
        model_override: Optional[str],
        max_output_tokens: Optional[int],
        locked_models: Optional[bool],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        await self.initialize()
        await self._emit_pending_ollama_issue(batch_id, claim_id, stage)

        messages = [
            {
                "role": "system",
                "content": system_prompt or (
                    load_prompt("gateway_default_json_system")
                    if expects_json
                    else load_prompt("gateway_default_text_system")
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        locked = self.routing_locked_models if locked_models is None else bool(locked_models)
        preflight = await self._route_and_preflight(
            task=task,
            messages=messages,
            model_override=model_override,
            max_output_tokens=max_output_tokens,
            locked_models=locked,
            batch_id=batch_id,
            claim_id=claim_id,
            paper_id=paper_id,
            stage=stage,
        )

        model_info = self.model_registry.get_model_info(preflight.model)
        response_format = {"type": "json_object"} if expects_json and model_info.supports_json_mode else None
        effective_temperature = temperature if model_info.supports_temperature else None
        request_template = {
            "model": preflight.model,
            "temperature": effective_temperature,
            "max_output_tokens": preflight.reserved_output_tokens,
            "response_format": response_format,
        }

        effective_timeout = self._resolve_timeout(task)

        parent_trace_id = None
        retry_count = 0
        json_recovery_used = False
        current_messages = list(messages)
        attempts_usage = empty_usage()

        for attempt in range(1, self.retry_policy.max_retries + 2):
            trace_id = str(uuid.uuid4())
            start_time = perf_counter()
            timestamp_start = utc_now_iso()
            finish_reason = None
            http_status = None
            parse_error = None
            raw_output = ""
            model_used = preflight.model
            usage = empty_usage(is_estimated=True)

            try:
                provider_request = ProviderRequest(
                    model=preflight.model,
                    messages=current_messages,
                    temperature=effective_temperature,
                    max_output_tokens=preflight.reserved_output_tokens,
                    response_format=response_format,
                    timeout_seconds=effective_timeout,
                )

                async with self.rate_limiter.reserve(preflight.estimated_total_tokens):
                    response = await asyncio.wait_for(
                        self.provider.chat(provider_request), timeout=effective_timeout
                    )

                raw_output = response.raw_text or ""
                model_used = response.model_used or preflight.model
                finish_reason = response.finish_reason
                http_status = response.http_status

                usage = self._finalize_usage(
                    usage=normalize_usage(response.usage),
                    model=model_used,
                    estimated_input_tokens=preflight.estimated_input_tokens,
                    raw_output=raw_output,
                    estimated=False,
                )
                attempts_usage = merge_usage(attempts_usage, usage)

                parsed_json = None
                if expects_json:
                    parsed_json, parse_error, recovered_json = self._parse_json_object_response(raw_output)
                    if parsed_json is None:
                        await self.add_issue(
                            batch_id=batch_id,
                            claim_id=claim_id,
                            severity="WARN",
                            stage=stage,
                            message="Invalid JSON returned by model.",
                            details={
                                "trace_id": trace_id,
                                "provider": self.provider_name,
                                "model": preflight.model,
                                "error": parse_error,
                            },
                        )
                        if not json_recovery_used:
                            json_recovery_used = True
                            await self._write_trace(
                                trace_id=trace_id,
                                parent_trace_id=parent_trace_id,
                                timestamp_start=timestamp_start,
                                start_time=start_time,
                                batch_id=batch_id,
                                claim_id=claim_id,
                                paper_id=paper_id,
                                stage=stage,
                                preflight=preflight,
                                request_template=request_template,
                                messages=current_messages,
                                raw_response=response.raw_response,
                                raw_output=raw_output,
                                parsed_json=None,
                                parse_error=parse_error,
                                usage=usage,
                                status="retrying",
                                finish_reason=finish_reason,
                                retries=retry_count,
                                timeout_configured_s=effective_timeout,
                                http_status=http_status,
                                error_type="InvalidJSONResponseError",
                                error_message="Invalid JSON returned by model. Retrying once with JSON reminder.",
                                stacktrace=None,
                                model_used=model_used,
                            )
                            parent_trace_id = trace_id
                            retry_count += 1
                            current_messages = current_messages + [
                                {
                                    "role": "user",
                                    "content": load_prompt("gateway_invalid_json_retry_user"),
                                }
                            ]
                            continue
                        raise InvalidJSONResponseError("Model returned invalid JSON after recovery attempt.")
                    if recovered_json:
                        await self.add_issue(
                            batch_id=batch_id,
                            claim_id=claim_id,
                            severity="WARN",
                            stage=stage,
                            message="Recovered JSON object from wrapped/non-compliant model output.",
                            details={
                                "trace_id": trace_id,
                                "provider": self.provider_name,
                                "model": preflight.model,
                            },
                        )

                await self.rate_limiter.adjust_usage(
                    estimated_tokens=preflight.estimated_total_tokens, actual_tokens=usage["total_tokens"]
                )
                await self._write_trace(
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    timestamp_start=timestamp_start,
                    start_time=start_time,
                    batch_id=batch_id,
                    claim_id=claim_id,
                    paper_id=paper_id,
                    stage=stage,
                    preflight=preflight,
                    request_template=request_template,
                    messages=current_messages,
                    raw_response=response.raw_response,
                    raw_output=raw_output,
                    parsed_json=parsed_json,
                    parse_error=parse_error,
                    usage=usage,
                    status="success",
                    finish_reason=finish_reason,
                    retries=retry_count,
                    timeout_configured_s=effective_timeout,
                    http_status=http_status,
                    error_type=None,
                    error_message=None,
                    stacktrace=None,
                    model_used=model_used,
                )

                if finish_reason == "length":
                    await self.add_issue(
                        batch_id=batch_id,
                        claim_id=claim_id,
                        severity="WARN",
                        stage=stage,
                        message="Model output reached token limit (finish_reason=length).",
                        details={
                            "trace_id": trace_id,
                            "model": model_used,
                        },
                    )

                return {
                    "content": parsed_json if expects_json else raw_output,
                    "usage": attempts_usage if attempts_usage.get("total_tokens", 0) > 0 else usage,
                    "model": model_used,
                    "trace_id": trace_id,
                }

            except Exception as exc:
                http_status = self._extract_http_status(exc)
                error_type = exc.__class__.__name__
                error_message = str(exc)
                elapsed_s = round(perf_counter() - start_time, 3)
                stacktrace = self._truncate_text(traceback.format_exc(), self.stacktrace_max_bytes)
                decision = self.retry_policy.classify(exc, http_status=http_status)
                should_retry = decision.should_retry and attempt <= self.retry_policy.max_retries

                # Pre-compute backoff so it can be recorded in trace
                backoff_s = (
                    self.retry_policy.compute_backoff_seconds(attempt)
                    if should_retry
                    else None
                )

                await self.add_issue(
                    batch_id=batch_id,
                    claim_id=claim_id,
                    severity="WARN" if should_retry else "ERROR",
                    stage=stage,
                    message=(
                        f"LLM call failed; retrying (attempt {attempt})."
                        if should_retry
                        else "LLM call failed after retries."
                    ),
                    details={
                        "trace_id": trace_id,
                        "provider": self.provider_name,
                        "model": preflight.model,
                        "http_status": http_status,
                        "exception_type": error_type,
                        "exception_message": error_message,
                        "timeout_configured_s": effective_timeout,
                        "elapsed_s": elapsed_s,
                    },
                )

                await self._write_trace(
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    timestamp_start=timestamp_start,
                    start_time=start_time,
                    batch_id=batch_id,
                    claim_id=claim_id,
                    paper_id=paper_id,
                    stage=stage,
                    preflight=preflight,
                    request_template=request_template,
                    messages=current_messages,
                    raw_response=None,
                    raw_output=raw_output,
                    parsed_json=None,
                    parse_error=parse_error,
                    usage=usage,
                    status="retrying" if should_retry else "error",
                    finish_reason=finish_reason,
                    retries=retry_count,
                    timeout_configured_s=effective_timeout,
                    backoff_waited_s=backoff_s,
                    http_status=http_status,
                    error_type=error_type,
                    error_message=error_message,
                    stacktrace=stacktrace,
                    model_used=model_used,
                )

                if should_retry:
                    parent_trace_id = trace_id
                    retry_count += 1
                    await asyncio.sleep(backoff_s)
                    continue
                raise

    async def _write_trace(
        self,
        *,
        trace_id: str,
        parent_trace_id: Optional[str],
        timestamp_start: str,
        start_time: float,
        batch_id: Optional[str],
        claim_id: Optional[str],
        paper_id: Optional[str],
        stage: str,
        preflight: PreflightResult,
        request_template: Dict[str, Any],
        messages: List[Dict[str, Any]],
        raw_response: Any,
        raw_output: str,
        parsed_json: Optional[Dict[str, Any]],
        parse_error: Optional[str],
        usage: Dict[str, Any],
        status: str,
        finish_reason: Optional[str],
        retries: int,
        timeout_configured_s: Optional[int] = None,
        backoff_waited_s: Optional[float] = None,
        http_status: Optional[int] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        stacktrace: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> None:
        timestamp_end = utc_now_iso()
        latency_ms = max(0, int((perf_counter() - start_time) * 1000))
        record = TraceRecord(
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            latency_ms=latency_ms,
            batch_id=batch_id,
            claim_id=claim_id,
            paper_id=paper_id,
            stage=stage,
            provider=self.provider_name,
            base_url=self._redact_base_url(self.base_url),
            model_requested=preflight.model,
            model_used=model_used or preflight.model,
            request=request_template,
            messages=messages,
            raw_response=self._truncate_object(raw_response),
            raw_output=self._truncate_text(raw_output, 2_000_000) or "",
            parsed_json=parsed_json,
            parse_error=parse_error,
            usage=usage,
            estimated_input_tokens=preflight.estimated_input_tokens,
            estimated_total_tokens=preflight.estimated_total_tokens,
            reserved_output_tokens=preflight.reserved_output_tokens,
            status=status,
            finish_reason=finish_reason,
            retries=retries,
            timeout_configured_s=timeout_configured_s,
            backoff_waited_s=backoff_waited_s,
            http_status=http_status,
            error_type=error_type,
            error_message=error_message,
            stacktrace=stacktrace,
            routing={"reason": preflight.reason, "context_window_tokens": preflight.context_window_tokens},
        ).to_dict()
        record["token_counts_are_estimated"] = bool(usage.get("is_estimated", False))
        record["cost_usd"] = float(usage.get("cost_usd", 0.0))
        await self.trace_store.append(batch_id, claim_id, record)

    async def _route_and_preflight(
        self,
        *,
        task: str,
        messages: List[Dict[str, Any]],
        model_override: Optional[str],
        max_output_tokens: Optional[int],
        locked_models: bool,
        batch_id: Optional[str],
        claim_id: Optional[str],
        paper_id: Optional[str],
        stage: str,
    ) -> PreflightResult:
        task_cfg = self._task_config(task)
        reserved_output_tokens = int(
            max_output_tokens
            or task_cfg.get("max_output_tokens")
            or self.model_registry.default_max_output_tokens(self.default_model)
        )
        candidates, base_reason = self._candidate_models(task_cfg, model_override)
        preferred_models = list(task_cfg.get("preferred_models", []))
        fallback_models = list(task_cfg.get("fallback_models", []))

        failures = []
        for model in candidates:
            est_input = self.token_estimator.estimate_chat_tokens(messages, model)
            context_window = self.model_registry.context_window(model)
            est_total = est_input + reserved_output_tokens + self.context_safety_margin_tokens
            if context_window and est_total <= context_window:
                reason = base_reason
                if not model_override and self.routing_enabled:
                    if model in preferred_models:
                        reason = "preferred fits"
                    elif model in fallback_models:
                        reason = "fallback due to context"
                return PreflightResult(
                    model=model,
                    estimated_input_tokens=est_input,
                    estimated_total_tokens=est_total,
                    reserved_output_tokens=reserved_output_tokens,
                    context_window_tokens=context_window,
                    reason=reason,
                )
            failures.append(
                {
                    "model": model,
                    "estimated_total_tokens": est_total,
                    "context_window_tokens": context_window,
                }
            )

        if locked_models and model_override:
            message = "Context overflow prevented: locked model does not fit prompt."
        else:
            message = "Context overflow prevented: no candidate model can fit prompt."

        await self.add_issue(
            batch_id=batch_id,
            claim_id=claim_id,
            severity="ERROR",
            stage=stage,
            message=message,
            details={
                "task": task,
                "paper_id": paper_id,
                "candidates": failures,
                "estimated_tokens": failures[0]["estimated_total_tokens"] if failures else 0,
                "context_limit": failures[0]["context_window_tokens"] if failures else 0,
            },
        )
        raise ContextOverflowError(message)

    def _task_config(self, task: str) -> Dict[str, Any]:
        merged = dict(DEFAULT_TASK_CONFIG.get(task, DEFAULT_TASK_CONFIG[LLMTask.GENERIC]))
        routing_tasks = self.routing_config.get("tasks", {})
        if task in routing_tasks:
            merged.update(routing_tasks[task] or {})
        if not merged.get("preferred_models"):
            merged["preferred_models"] = [self.default_model]
        if merged.get("fallback_models") is None:
            merged["fallback_models"] = []
        return merged

    def _is_local_provider(self) -> bool:
        return self.provider_name in {"local", "llamacpp", "ollama"}

    def _resolve_timeout(self, task: str) -> int:
        """Resolve timeout: task-level override > local provider default > global default."""
        task_cfg = self._task_config(task)
        task_timeout = task_cfg.get("timeout_seconds")
        if task_timeout is not None:
            return int(task_timeout)
        if self._is_local_provider() and self.timeout_seconds_local is not None:
            return int(self.timeout_seconds_local)
        return self.timeout_seconds

    def _candidate_models(self, task_config: Dict[str, Any], model_override: Optional[str]) -> Tuple[List[str], str]:
        if model_override:
            return [model_override], "override"
        if not self.routing_enabled:
            return [self.default_model], "routing_disabled"
        preferred = list(task_config.get("preferred_models", []))
        fallback = list(task_config.get("fallback_models", []))
        seen = set()
        models = []
        for model in preferred + fallback:
            if model and model not in seen:
                seen.add(model)
                models.append(model)
        if not models:
            models = [self.default_model]
        return models, "preferred_then_fallback"

    def _finalize_usage(
        self,
        *,
        usage: Dict[str, Any],
        model: str,
        estimated_input_tokens: int,
        raw_output: str,
        estimated: bool,
    ) -> Dict[str, Any]:
        normalized = normalize_usage(usage)
        input_tokens = normalized["input_tokens"] or estimated_input_tokens
        output_tokens = normalized["output_tokens"] or self.token_estimator.estimate_text_tokens(raw_output, model)
        total_tokens = normalized["total_tokens"] or (input_tokens + output_tokens)
        cost_usd = normalized["cost_usd"]
        if cost_usd == 0 and self.provider_name not in {"local", "llamacpp", "ollama"}:
            cost_usd = self.model_registry.calculate_cost(model, input_tokens, output_tokens)
        if self.provider_name in {"local", "llamacpp", "ollama"}:
            cost_usd = 0.0
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "cost_usd": float(cost_usd),
            "is_estimated": bool(normalized["is_estimated"] or estimated),
        }

    def _build_provider(self):
        if self.provider_name == "azure-openai":
            return AzureOpenAIProvider(
                api_key=Config.LLM_API_KEY,
                endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION,
            )
        if self.provider_name == "azure-inference":
            return AzureInferenceProvider(
                endpoint=Config.AZURE_AI_INFERENCE_ENDPOINT,
                api_key=Config.LLM_API_KEY,
            )
        if self.provider_name == "openrouter":
            base_url = self.base_url or "https://openrouter.ai/api/v1"
            extra_headers = {}
            if getattr(Config, "LLM_HTTP_REFERER", None):
                extra_headers["HTTP-Referer"] = Config.LLM_HTTP_REFERER
            if getattr(Config, "LLM_SITE_NAME", None):
                extra_headers["X-Title"] = Config.LLM_SITE_NAME
            return OpenRouterProvider(
                api_key=Config.LLM_API_KEY,
                base_url=base_url,
                extra_headers=extra_headers,
            )
        if self.provider_name == "llamacpp":
            return LlamaCppProvider(base_url=self.base_url)
        if self.provider_name == "local":
            if self.local_backend == "ollama":
                return OllamaProvider(base_url=self.base_url)
            if self.local_backend == "llamacpp":
                return LlamaCppProvider(base_url=self.base_url)
            return OpenAICompatibleProvider(api_key="sk-no-key-required", base_url=self.base_url)
        if self.provider_name == "ollama":
            return OllamaProvider(base_url=self.base_url)
        if self.base_url and self.base_url != "http://localhost:8000":
            return OpenAICompatibleProvider(api_key=Config.LLM_API_KEY, base_url=self.base_url)
        return OpenAIProvider(api_key=Config.LLM_API_KEY)

    def _is_ollama_backend(self) -> bool:
        if self.provider_name == "ollama":
            return True
        if self.provider_name == "local" and self.local_backend == "ollama":
            return True
        return bool(self.base_url and "11434" in self.base_url and self.provider_name in {"local", "openai"})

    async def _initialize_ollama_context(self) -> None:
        show_url = getattr(Config, "OLLAMA_SHOW_URL", None) or self._derive_ollama_show_url(self.base_url)
        model_name = self.default_model
        try:
            details = None
            if isinstance(self.provider, OllamaProvider):
                details = await self.provider.fetch_model_details(model_name=model_name, show_url=show_url)
            else:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(show_url, json={"model": model_name})
                    response.raise_for_status()
                    details = response.json()
            if not isinstance(details, dict):
                raise RuntimeError("Ollama show API returned invalid payload.")

            num_ctx = self._extract_num_ctx(details.get("parameters"))
            model_context = self._extract_context_length(details.get("model_info", {}))
            effective_context = self._compute_effective_context(num_ctx, model_context)
            if effective_context:
                self.model_registry.set_runtime_context_window(model_name, effective_context)
                self._ollama_show_error = None
            else:
                self._ollama_show_error = "Ollama model context could not be determined from /api/show response."
        except Exception as exc:
            self._ollama_show_error = f"Ollama model context introspection failed: {exc}"
            logger.warning(self._ollama_show_error)

    async def _emit_pending_ollama_issue(
        self, batch_id: Optional[str], claim_id: Optional[str], stage: str
    ) -> None:
        if not self._ollama_show_error or not batch_id or not claim_id:
            return
        key = (batch_id, claim_id)
        if key in self._ollama_warning_claims:
            return
        self._ollama_warning_claims.add(key)
        await self.add_issue(
            batch_id=batch_id,
            claim_id=claim_id,
            severity="WARN",
            stage=stage,
            message=self._ollama_show_error,
            details={
                "provider": self.provider_name,
                "model": self.default_model,
                "guidance": (
                    "For Ollama, create a model with larger `num_ctx` in a Modelfile and call that model name."
                ),
            },
        )

    async def _maybe_write_trace_capped_issue(self, batch_id: str, claim_id: str) -> None:
        issues = await self.get_claim_issues(batch_id, claim_id)
        marker = "Trace embedding capped"
        if any(marker in issue.get("message", "") for issue in issues):
            return
        await self.add_issue(
            batch_id=batch_id,
            claim_id=claim_id,
            severity="WARN",
            stage="system",
            message="Trace embedding capped; full trace is available in trace file.",
            details={"trace_file": f"traces/{claim_id}.jsonl"},
        )

    def _build_trace_summary(self, calls: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        usage = empty_usage()
        models_used = set()
        retries = 0
        for call in calls:
            models_used.add(call.get("model_used") or call.get("model_requested"))
            usage = merge_usage(usage, call.get("usage"))
            if call.get("status") == "retrying":
                retries += 1
        context_overflow_prevented = sum(
            1 for issue in issues if "Context overflow prevented" in (issue.get("message") or "")
        )
        intentional_truncations = sum(
            1 for issue in issues if "truncat" in (issue.get("message") or "").lower()
        )
        return {
            "llm_calls": len(calls),
            "models_used": sorted([m for m in models_used if m]),
            "total_input_tokens": usage["input_tokens"],
            "total_output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "total_cost_usd": usage["cost_usd"],
            "retries": retries,
            "context_overflow_prevented": context_overflow_prevented,
            "intentional_truncations": intentional_truncations,
        }

    def _cap_trace_calls(self, calls: List[Dict[str, Any]], max_bytes: int) -> Tuple[List[Dict[str, Any]], bool]:
        embedded = []
        bytes_used = 0
        capped = False
        for record in calls:
            serialized = json.dumps(record, ensure_ascii=True)
            size = len(serialized.encode("utf-8"))
            if bytes_used + size > max_bytes:
                capped = True
                break
            embedded.append(record)
            bytes_used += size
        return embedded, capped

    @staticmethod
    def _derive_ollama_show_url(base_url: Optional[str]) -> str:
        if not base_url:
            return "http://localhost:11434/api/show"
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            trimmed = trimmed[:-3]
        return f"{trimmed}/api/show"

    @staticmethod
    def _extract_num_ctx(parameters: Any) -> Optional[int]:
        if isinstance(parameters, str):
            match = re.search(r"num_ctx\\s+(\\d+)", parameters)
            if match:
                return int(match.group(1))
        if isinstance(parameters, dict):
            value = parameters.get("num_ctx")
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    return None
        return None

    @classmethod
    def _extract_context_length(cls, model_info: Any) -> Optional[int]:
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if key.endswith("context_length"):
                    try:
                        return int(value)
                    except Exception:
                        continue
                nested = cls._extract_context_length(value)
                if nested:
                    return nested
        if isinstance(model_info, list):
            for item in model_info:
                nested = cls._extract_context_length(item)
                if nested:
                    return nested
        return None

    @staticmethod
    def _compute_effective_context(num_ctx: Optional[int], model_context: Optional[int]) -> Optional[int]:
        values = [v for v in [num_ctx, model_context] if isinstance(v, int) and v > 0]
        if not values:
            return None
        return min(values)

    @staticmethod
    def _extract_http_status(exc: Exception) -> Optional[int]:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        if response is not None:
            code = getattr(response, "status_code", None)
            if isinstance(code, int):
                return code
        return None

    @staticmethod
    def _redact_base_url(base_url: Optional[str]) -> Optional[str]:
        if not base_url:
            return None
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            return base_url
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _truncate_object(data: Any, max_chars: int = 50_000) -> Any:
        if data is None:
            return None
        try:
            text = json.dumps(data, ensure_ascii=True)
            if len(text) <= max_chars:
                return data
            return {"_truncated": True, "preview": text[:max_chars]}
        except Exception:
            return str(data)[:max_chars]

    @staticmethod
    def _truncate_text(text: Optional[str], max_chars: int) -> Optional[str]:
        if text is None:
            return None
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"

    def _parse_json_object_response(self, raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], bool]:
        strict_error = None
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed, None, False
            return None, f"Expected JSON object, got {type(parsed).__name__}.", False
        except Exception as exc:
            strict_error = str(exc)

        for candidate in self._json_object_candidates(raw_output):
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed, None, True

        return None, strict_error or "Invalid JSON response.", False

    @classmethod
    def _json_object_candidates(cls, text: str) -> List[str]:
        candidates: List[str] = []

        fenced_matches = re.findall(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        for match in fenced_matches:
            if isinstance(match, str) and match.strip():
                candidates.append(match.strip())

        first_object = cls._extract_first_balanced_object(text)
        if first_object:
            candidates.append(first_object)

        # Preserve order while removing duplicates.
        seen = set()
        deduped = []
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _extract_first_balanced_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None

    @staticmethod
    def _iso_to_epoch(value: str) -> float:
        return datetime.fromisoformat(value).timestamp()
