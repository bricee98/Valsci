from app.config.settings import Config
from app.services.evidence_scorer import EvidenceScorer


def test_evidence_scorer_uses_dynamic_current_year(monkeypatch):
    monkeypatch.setattr(EvidenceScorer, "_current_year", staticmethod(lambda: 2026))
    scorer = EvidenceScorer()

    assert scorer._calculate_citation_impact({"citationCount": 24, "year": 2024}) == 12
    assert scorer._calculate_citation_impact({"citationCount": 24, "year": 2030}) == 24


def test_config_report_redacts_values_and_calls_out_defaults(monkeypatch):
    metadata = {
        "LLM_API_KEY": {
            "env_key": "LLM_API_KEY",
            "source": "env_vars.json",
            "reason": "provided",
            "value": "super-secret",
        },
        "SMTP_SERVER": {
            "env_key": "SMTP_SERVER",
            "source": "default",
            "reason": "missing",
            "default_value": "smtp.gmail.com",
            "value": "smtp.gmail.com",
        },
        "SMTP_PORT": {
            "env_key": "SMTP_PORT",
            "source": "fallback_default",
            "reason": "invalid",
            "default_value": 587,
            "raw_value": "abc",
            "value": 587,
        },
        "EMAIL_SENDER": {
            "env_key": "EMAIL_SENDER",
            "source": "unset",
            "reason": "missing",
            "value": None,
        },
    }

    monkeypatch.setattr(Config, "_CONFIG_METADATA", metadata, raising=False)
    monkeypatch.setattr(Config, "_CONFIG_REPORT_EMITTED", False, raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "super-secret", raising=False)
    monkeypatch.setattr(Config, "SMTP_SERVER", "smtp.gmail.com", raising=False)
    monkeypatch.setattr(Config, "SMTP_PORT", 587, raising=False)
    monkeypatch.setattr(Config, "EMAIL_SENDER", None, raising=False)

    entries = {entry["key"]: entry for entry in Config.get_effective_config_entries()}

    assert entries["LLM_API_KEY"]["value"] == "<redacted>"
    assert entries["SMTP_SERVER"]["note"] == "no custom value provided; using default 'smtp.gmail.com'."
    assert entries["SMTP_PORT"]["note"] == "custom value 'abc' is invalid; using default 587."
    assert entries["EMAIL_SENDER"]["note"] == "no custom value provided and no default is configured."

    lines = []
    Config.emit_config_report(printer=lines.append, force=True)

    assert lines[0] == "Effective config:"
    assert any("LLM_API_KEY = <redacted> [env_vars.json]" in line for line in lines)
    assert any("Config default used for SMTP_SERVER:" in line for line in lines)
    assert any("Config fallback default used for SMTP_PORT:" in line for line in lines)
    assert any("Config value unset for EMAIL_SENDER:" in line for line in lines)
