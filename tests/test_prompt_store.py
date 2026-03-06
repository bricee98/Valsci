import pytest

from app.services.prompt_store import load_prompt, render_prompt


def test_prompt_files_load():
    assert load_prompt("query_generation_system")
    assert load_prompt("query_generation_user")
    assert load_prompt("paper_analysis_system")
    assert load_prompt("paper_analysis_user")
    assert load_prompt("venue_scoring_system")
    assert load_prompt("venue_scoring_user")
    assert load_prompt("final_report_system")
    assert load_prompt("final_report_user")
    assert load_prompt("gateway_default_json_system")
    assert load_prompt("gateway_default_text_system")
    assert load_prompt("gateway_invalid_json_retry_user")


def test_render_prompt_replaces_placeholders():
    text = render_prompt(
        "query_generation_user",
        num_queries=3,
        claim_text="Creatine improves cognition",
    )
    assert "Generate 3 strategic search queries" in text
    assert "Creatine improves cognition" in text


def test_render_prompt_raises_on_missing_placeholder():
    with pytest.raises(ValueError):
        render_prompt("query_generation_user", num_queries=2)
