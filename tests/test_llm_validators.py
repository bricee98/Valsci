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


def test_validate_query_generation_payload_truncates_extra_explanations():
    payload = {
        "explanations": [
            "reason one",
            "reason two",
            "reason three",
        ],
        "queries": ["query one", "query two"],
    }

    validated = validate_query_generation_payload(payload, expected_query_count=2)

    assert validated["queries"] == payload["queries"]
    assert validated["explanations"] == ["reason one", "reason two"]


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


def test_validate_final_report_coerces_nonstring_reasoning():
    # Reasoning models sometimes return finalReasoning as a list of points or an
    # object; coerce to text rather than failing the whole report.
    list_payload = {
        "explanationEssay": "Detailed explanation.",
        "finalReasoning": ["Point one.", "Point two."],
        "claimRating": "Likely True",
    }
    validated = validate_final_report_payload(list_payload)
    assert "Point one." in validated["finalReasoning"]
    assert "Point two." in validated["finalReasoning"]

    dict_payload = {
        "explanationEssay": {"summary": "essay as object"},
        "finalReasoning": "ok",
        "claimRating": "Mixed Evidence",
    }
    validated = validate_final_report_payload(dict_payload)
    assert "essay as object" in validated["explanationEssay"]
    assert validated["claimRating"] == "Mixed Evidence"


def test_validate_final_report_still_rejects_nonstring_rating():
    # claimRating must stay a recognizable verdict — coercion does not apply.
    import pytest
    from app.services.llm.validators import OutputValidationError

    payload = {
        "explanationEssay": "x",
        "finalReasoning": "y",
        "claimRating": {"verdict": "Likely True"},
    }
    with pytest.raises(OutputValidationError):
        validate_final_report_payload(payload)
