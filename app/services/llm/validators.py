"""Validation helpers for untrusted LLM outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class OutputValidationError(ValueError):
    """Raised when an LLM response does not match the expected schema."""


_CLAIM_RATING_CANONICAL = {
    "contradicted": "Contradicted",
    "likely false": "Likely False",
    "mixed evidence": "Mixed Evidence",
    "likely true": "Likely True",
    "highly supported": "Highly Supported",
    "no evidence": "No Evidence",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _strip_list_prefix(text: str) -> str:
    return re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", text).strip()


def _require_string(value: Any, field_name: str, *, max_len: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise OutputValidationError(f"Field '{field_name}' must be a string.")
    cleaned = _strip_code_fence(value).strip().strip("`")
    cleaned = cleaned.replace("\r", "\n").replace("\n", " ")
    cleaned = _normalize_whitespace(cleaned)
    if not cleaned and not allow_empty:
        raise OutputValidationError(f"Field '{field_name}' must not be empty.")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise OutputValidationError(f"Field '{field_name}' must be numeric.") from exc


def sanitize_query(query: str, *, max_len: int = 220) -> str:
    cleaned = _require_string(query, "query", max_len=max_len * 2)
    cleaned = _strip_list_prefix(cleaned)
    if ":" in cleaned and cleaned.lower().startswith("query"):
        cleaned = cleaned.split(":", 1)[1].strip()
    cleaned = cleaned.strip("\"'")
    cleaned = _normalize_whitespace(cleaned)
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (cleaned.startswith("[") and cleaned.endswith("]")):
        raise OutputValidationError("Query contains structured JSON instead of plain text.")
    if not cleaned:
        raise OutputValidationError("Query must not be empty.")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def validate_query_list(
    queries: Any,
    *,
    expected_count: Optional[int] = None,
    min_count: int = 1,
    max_count: int = 25,
    max_query_len: int = 220,
) -> List[str]:
    if not isinstance(queries, list):
        raise OutputValidationError("Field 'queries' must be a list of strings.")

    cleaned_queries: List[str] = []
    seen = set()
    for item in queries:
        if not isinstance(item, str):
            raise OutputValidationError("Every query must be a string.")
        cleaned = sanitize_query(item, max_len=max_query_len)
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_queries.append(cleaned)
        if len(cleaned_queries) >= max_count:
            break

    if expected_count is not None and expected_count > 0 and len(cleaned_queries) != expected_count:
        raise OutputValidationError(
            f"Expected exactly {expected_count} unique queries, got {len(cleaned_queries)}."
        )
    if len(cleaned_queries) < min_count:
        raise OutputValidationError(f"Expected at least {min_count} query entries.")
    return cleaned_queries


def validate_query_generation_payload(payload: Any, *, expected_query_count: int) -> Dict[str, List[str]]:
    if not isinstance(payload, dict):
        raise OutputValidationError("Query generation output must be a JSON object.")

    queries = validate_query_list(
        payload.get("queries"),
        expected_count=expected_query_count,
        min_count=expected_query_count if expected_query_count > 0 else 1,
        max_count=max(1, expected_query_count or 25),
    )

    raw_explanations = payload.get("explanations")
    if not isinstance(raw_explanations, list):
        raise OutputValidationError("Field 'explanations' must be a list of strings.")
    explanations = [
        _require_string(text, f"explanations[{idx}]", max_len=1200)
        for idx, text in enumerate(raw_explanations)
    ]
    if expected_query_count > 0:
        explanations = explanations[:expected_query_count]
    if expected_query_count > 0 and len(explanations) < expected_query_count:
        raise OutputValidationError(
            f"Expected at least {expected_query_count} explanations, got {len(explanations)}."
        )

    return {
        "queries": queries,
        "explanations": explanations,
    }


def validate_paper_analysis_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise OutputValidationError("Paper analysis output must be a JSON object.")

    relevance = _coerce_float(payload.get("relevance", 0), "relevance")
    relevance = max(0.0, min(1.0, relevance))

    raw_excerpts = payload.get("excerpts", [])
    if not isinstance(raw_excerpts, list):
        raise OutputValidationError("Field 'excerpts' must be a list of strings.")
    excerpts = [
        _require_string(item, f"excerpts[{idx}]", max_len=3000)
        for idx, item in enumerate(raw_excerpts)
    ]

    raw_explanations = payload.get("explanations", [])
    if not isinstance(raw_explanations, list):
        raise OutputValidationError("Field 'explanations' must be a list of strings.")
    explanations = [
        _require_string(item, f"explanations[{idx}]", max_len=1800)
        for idx, item in enumerate(raw_explanations)
    ]

    if len(explanations) < len(excerpts):
        explanations.extend([""] * (len(excerpts) - len(explanations)))
    elif len(explanations) > len(excerpts):
        explanations = explanations[: len(excerpts)]

    raw_pages = payload.get("excerpt_pages", [])
    if raw_pages is None:
        raw_pages = []
    if not isinstance(raw_pages, list):
        raise OutputValidationError("Field 'excerpt_pages' must be a list.")
    pages: List[Optional[int]] = []
    for item in raw_pages[: len(excerpts)]:
        if item is None:
            pages.append(None)
            continue
        try:
            parsed = int(item)
        except Exception:
            pages.append(None)
            continue
        pages.append(parsed if parsed > 0 else None)
    if len(pages) < len(excerpts):
        pages.extend([None] * (len(excerpts) - len(pages)))

    non_relevant_explanation_raw = payload.get("non_relevant_explanation")
    non_relevant_explanation = None
    if non_relevant_explanation_raw is not None:
        non_relevant_explanation = _require_string(
            non_relevant_explanation_raw,
            "non_relevant_explanation",
            max_len=2000,
            allow_empty=True,
        )

    return {
        "relevance": relevance,
        "excerpts": excerpts,
        "explanations": explanations,
        "non_relevant_explanation": non_relevant_explanation,
        "excerpt_pages": pages,
    }


def validate_venue_score_payload(payload: Any) -> Dict[str, float]:
    if not isinstance(payload, dict):
        raise OutputValidationError("Venue scoring output must be a JSON object.")
    score = _coerce_float(payload.get("score"), "score")
    score = max(0.0, min(10.0, score))
    return {"score": score}


def validate_final_report_payload(payload: Any) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise OutputValidationError("Final report output must be a JSON object.")

    explanation = _require_string(payload.get("explanationEssay"), "explanationEssay", max_len=20_000)
    reasoning = _require_string(payload.get("finalReasoning"), "finalReasoning", max_len=20_000)
    raw_rating = _require_string(payload.get("claimRating"), "claimRating", max_len=128)
    canonical = _CLAIM_RATING_CANONICAL.get(raw_rating.lower())
    if not canonical:
        raise OutputValidationError(
            "Field 'claimRating' must be one of: Contradicted, Likely False, Mixed Evidence, Likely True, Highly Supported, No Evidence."
        )

    return {
        "explanationEssay": explanation,
        "finalReasoning": reasoning,
        "claimRating": canonical,
    }
