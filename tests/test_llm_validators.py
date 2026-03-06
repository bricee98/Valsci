import pytest

from app.services.llm.validators import (
    OutputValidationError,
    validate_final_report_payload,
    validate_query_generation_payload,
    validate_query_list,
)


def test_validate_query_generation_payload_accepts_expected_shape():
    payload = {
        "explanations": ["explanation one", "explanation two"],
        "queries": ["insulin sensitivity aging", "metformin longevity trial"],
    }
    validated = validate_query_generation_payload(payload, expected_query_count=2)
    assert validated["queries"] == payload["queries"]
    assert validated["explanations"] == payload["explanations"]


def test_validate_query_generation_payload_rejects_wrong_count():
    payload = {
        "explanations": ["why this query"],
        "queries": ["query one"],
    }
    with pytest.raises(OutputValidationError):
        validate_query_generation_payload(payload, expected_query_count=2)


def test_validate_query_list_sanitizes_wrapped_lines():
    queries = ["1. Query: insulin resistance aging", "- metformin mortality cohort study"]
    validated = validate_query_list(queries, expected_count=2)
    assert validated == ["insulin resistance aging", "metformin mortality cohort study"]


def test_validate_final_report_payload_normalizes_rating_case():
    payload = {
        "explanationEssay": "Detailed explanation.",
        "finalReasoning": "Additional analysis.",
        "claimRating": "likely true",
    }
    validated = validate_final_report_payload(payload)
    assert validated["claimRating"] == "Likely True"
