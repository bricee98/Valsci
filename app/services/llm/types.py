"""Core types shared by LLM gateway components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    is_estimated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def empty_usage(is_estimated: bool = False) -> Dict[str, Any]:
    return UsageStats(is_estimated=is_estimated).to_dict()


def normalize_usage(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return empty_usage()
    input_tokens = int(data.get("input_tokens", data.get("prompt_tokens", 0)) or 0)
    output_tokens = int(data.get("output_tokens", data.get("completion_tokens", 0)) or 0)
    total_tokens = int(data.get("total_tokens", input_tokens + output_tokens) or 0)
    cost_usd = float(data.get("cost_usd", data.get("cost", data.get("total_cost", 0.0))) or 0.0)
    is_estimated = bool(data.get("is_estimated", False))
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        is_estimated=is_estimated,
    ).to_dict()


def merge_usage(base: Optional[Dict[str, Any]], add: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base_usage = normalize_usage(base)
    add_usage = normalize_usage(add)
    merged = UsageStats(
        input_tokens=base_usage["input_tokens"] + add_usage["input_tokens"],
        output_tokens=base_usage["output_tokens"] + add_usage["output_tokens"],
        total_tokens=base_usage["total_tokens"] + add_usage["total_tokens"],
        cost_usd=round(base_usage["cost_usd"] + add_usage["cost_usd"], 10),
        is_estimated=base_usage["is_estimated"] or add_usage["is_estimated"],
    )
    return merged.to_dict()


@dataclass
class ProviderRequest:
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 180
    extra_headers: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderResponse:
    raw_text: str
    model_used: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    http_status: Optional[int] = None
    raw_response: Any = None


@dataclass
class TraceRecord:
    trace_id: str
    parent_trace_id: Optional[str]
    timestamp_start: str
    timestamp_end: str
    latency_ms: int
    batch_id: Optional[str]
    claim_id: Optional[str]
    paper_id: Optional[str]
    stage: str
    provider: str
    base_url: Optional[str]
    model_requested: str
    model_used: Optional[str]
    request: Dict[str, Any]
    messages: List[Dict[str, Any]]
    raw_response: Any
    raw_output: str
    parsed_json: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    usage: Dict[str, Any]
    estimated_input_tokens: int
    estimated_total_tokens: int
    reserved_output_tokens: int
    status: str
    finish_reason: Optional[str]
    retries: int
    timeout_configured_s: Optional[int]
    timeout_source: Optional[str]
    timed_out: bool
    backoff_waited_s: Optional[float]
    http_status: Optional[int]
    error_type: Optional[str]
    error_message: Optional[str]
    stacktrace: Optional[str]
    routing: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IssueRecord:
    issue_id: str
    timestamp: str
    severity: str
    stage: str
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
