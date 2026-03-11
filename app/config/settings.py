import json
import os
from pathlib import Path
from typing import Any, Dict


env_file_path = Path(__file__).parent.parent / "config/env_vars.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MISSING = object()
_SENSITIVE_KEYS = {
    "SECRET_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "LLM_API_KEY",
    "EMAIL_APP_PASSWORD",
    "ACCESS_PASSWORD",
}
_CONFIG_METADATA: Dict[str, Dict[str, Any]] = {}


try:
    with open(env_file_path, "r", encoding="utf-8") as f:
        env_vars = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"env_vars.json not found. Please create it at {env_file_path}")
except json.JSONDecodeError:
    raise ValueError("env_vars.json is not a valid JSON file")


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clone_value(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _resolve_project_path(value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _record_config(
    name,
    env_key,
    value,
    source,
    *,
    default_value=_MISSING,
    raw_value=_MISSING,
    reason="provided",
):
    metadata = {
        "env_key": env_key,
        "source": source,
        "reason": reason,
        "value": _clone_value(value),
    }
    if default_value is not _MISSING:
        metadata["default_value"] = _clone_value(default_value)
    if raw_value is not _MISSING:
        metadata["raw_value"] = _clone_value(raw_value)
    _CONFIG_METADATA[name] = metadata
    return value


def _string_setting(name, default=_MISSING, env_key=None):
    env_name = env_key or name
    if env_name in env_vars:
        raw_value = env_vars.get(env_name)
        return _record_config(name, env_name, raw_value, "env_vars.json", raw_value=raw_value)
    if default is _MISSING:
        return _record_config(name, env_name, None, "unset", reason="missing")
    return _record_config(name, env_name, default, "default", default_value=default, reason="missing")


def _bool_setting(name, default=False, env_key=None):
    env_name = env_key or name
    if env_name not in env_vars:
        return _record_config(name, env_name, bool(default), "default", default_value=default, reason="missing")
    raw_value = env_vars.get(env_name)
    if raw_value is None:
        return _record_config(
            name,
            env_name,
            bool(default),
            "fallback_default",
            default_value=default,
            raw_value=raw_value,
            reason="null",
        )
    return _record_config(
        name,
        env_name,
        _as_bool(raw_value, default=default),
        "env_vars.json",
        raw_value=raw_value,
    )


def _int_setting(name, default=0, env_key=None):
    env_name = env_key or name
    if env_name not in env_vars:
        return _record_config(name, env_name, default, "default", default_value=default, reason="missing")
    raw_value = env_vars.get(env_name)
    try:
        value = int(raw_value)
    except Exception:
        return _record_config(
            name,
            env_name,
            default,
            "fallback_default",
            default_value=default,
            raw_value=raw_value,
            reason="invalid",
        )
    return _record_config(name, env_name, value, "env_vars.json", raw_value=raw_value)


def _float_setting(name, default=0.0, env_key=None):
    env_name = env_key or name
    if env_name not in env_vars:
        return _record_config(name, env_name, default, "default", default_value=default, reason="missing")
    raw_value = env_vars.get(env_name)
    try:
        value = float(raw_value)
    except Exception:
        return _record_config(
            name,
            env_name,
            default,
            "fallback_default",
            default_value=default,
            raw_value=raw_value,
            reason="invalid",
        )
    return _record_config(name, env_name, value, "env_vars.json", raw_value=raw_value)


def _dict_setting(name, default=None, env_key=None):
    env_name = env_key or name
    default_value = {} if default is None else _clone_value(default)
    if env_name not in env_vars:
        return _record_config(
            name,
            env_name,
            _clone_value(default_value),
            "default",
            default_value=default_value,
            reason="missing",
        )

    raw_value = env_vars.get(env_name)
    if isinstance(raw_value, dict):
        return _record_config(name, env_name, dict(raw_value), "env_vars.json", raw_value=raw_value)
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return _record_config(name, env_name, parsed, "env_vars.json", raw_value=raw_value)

    return _record_config(
        name,
        env_name,
        _clone_value(default_value),
        "fallback_default",
        default_value=default_value,
        raw_value=raw_value,
        reason="invalid",
    )


def _path_setting(name, default, env_key=None):
    env_name = env_key or name
    if env_name in env_vars:
        raw_value = env_vars.get(env_name)
        if raw_value is not None and (not isinstance(raw_value, str) or raw_value.strip()):
            return _record_config(
                name,
                env_name,
                _resolve_project_path(raw_value),
                "env_vars.json",
                raw_value=raw_value,
            )
        return _record_config(
            name,
            env_name,
            _resolve_project_path(default),
            "default",
            default_value=default,
            raw_value=raw_value,
            reason="blank",
        )

    return _record_config(
        name,
        env_name,
        _resolve_project_path(default),
        "default",
        default_value=default,
        reason="missing",
    )


def _format_config_value(name, value, *, redact=True):
    if redact and name in _SENSITIVE_KEYS:
        if value in (None, ""):
            return "<empty>"
        return "<redacted>"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return repr(value)


class Config:
    PROJECT_ROOT = str(PROJECT_ROOT.resolve())
    SECRET_KEY = _string_setting("SECRET_KEY", env_key="FLASK_SECRET_KEY")
    SEMANTIC_SCHOLAR_API_KEY = _string_setting("SEMANTIC_SCHOLAR_API_KEY")
    USER_EMAIL = _string_setting("USER_EMAIL")

    AZURE_OPENAI_ENDPOINT = _string_setting("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = _string_setting("AZURE_OPENAI_API_VERSION", default="2024-06-01")

    AZURE_AI_INFERENCE_ENDPOINT = _string_setting("AZURE_AI_INFERENCE_ENDPOINT")

    ENABLE_EMAIL_NOTIFICATIONS = _bool_setting("ENABLE_EMAIL_NOTIFICATIONS", default=False)
    EMAIL_SENDER = _string_setting("EMAIL_SENDER")
    EMAIL_APP_PASSWORD = _string_setting("EMAIL_APP_PASSWORD")
    SMTP_SERVER = _string_setting("SMTP_SERVER", default="smtp.gmail.com")
    SMTP_PORT = _int_setting("SMTP_PORT", default=587)
    BASE_URL = _string_setting("BASE_URL", default="NO URL SET")

    REQUIRE_PASSWORD = _bool_setting("REQUIRE_PASSWORD", default=False)
    ACCESS_PASSWORD = _string_setting("ACCESS_PASSWORD")

    LLM_PROVIDER = _string_setting("LLM_PROVIDER", default="openai")
    LLM_BASE_URL = _string_setting("LLM_BASE_URL", default="http://localhost:8000")
    LLM_API_KEY = _string_setting("LLM_API_KEY", default="")
    LLM_EVALUATION_MODEL = _string_setting("LLM_EVALUATION_MODEL", default="gpt-4o")
    LLM_HTTP_REFERER = _string_setting("LLM_HTTP_REFERER")
    LLM_SITE_NAME = _string_setting("LLM_SITE_NAME")
    LOCAL_BACKEND = _string_setting("LOCAL_BACKEND", default="")
    LOCAL_MODEL_CONTEXT_OVERRIDE = _int_setting("LOCAL_MODEL_CONTEXT_OVERRIDE", default=0)
    if LOCAL_MODEL_CONTEXT_OVERRIDE == 0:
        LOCAL_MODEL_CONTEXT_OVERRIDE = None
        _CONFIG_METADATA["LOCAL_MODEL_CONTEXT_OVERRIDE"]["value"] = None

    RATE_LIMIT_MAX_TOKENS_PER_CLAIM = _int_setting("RATE_LIMIT_MAX_TOKENS_PER_CLAIM", default=300000)
    RATE_LIMIT_MAX_TOKENS_PER_WINDOW = _int_setting("RATE_LIMIT_MAX_TOKENS_PER_WINDOW", default=25000)
    RATE_LIMIT_MAX_REQUESTS_PER_WINDOW = _int_setting("RATE_LIMIT_MAX_REQUESTS_PER_WINDOW", default=5)
    RATE_LIMIT_WINDOW_SIZE_SECONDS = _int_setting("RATE_LIMIT_WINDOW_SIZE_SECONDS", default=10)

    TRACE_ENABLED = _bool_setting("TRACE_ENABLED", default=True)
    SAVED_JOBS_DIR = _path_setting("SAVED_JOBS_DIR", "saved_jobs")
    QUEUED_JOBS_DIR = _path_setting("QUEUED_JOBS_DIR", "queued_jobs")
    STATE_DIR = _path_setting("STATE_DIR", "state")
    PROVIDER_CATALOG_PATH = _path_setting(
        "PROVIDER_CATALOG_PATH",
        os.path.join(STATE_DIR, "provider_catalog.json"),
    )
    TRACE_DIR = _path_setting("TRACE_DIR", SAVED_JOBS_DIR)
    TRACE_EMBED_MODE = _string_setting("TRACE_EMBED_MODE", default="capped")
    TRACE_EMBED_MAX_BYTES = _int_setting("TRACE_EMBED_MAX_BYTES", default=2_000_000)
    TRACE_STACKTRACE_MAX_BYTES = _int_setting("TRACE_STACKTRACE_MAX_BYTES", default=4_000)
    TRACE_ALWAYS_WRITE_FILES = _bool_setting("TRACE_ALWAYS_WRITE_FILES", default=True)
    TRACE_COMPRESS_ON_COMPLETE = _bool_setting("TRACE_COMPRESS_ON_COMPLETE", default=False)

    LLM_ROUTING = _dict_setting("LLM_ROUTING", default={})
    MODEL_REGISTRY_OVERRIDES = _dict_setting("MODEL_REGISTRY_OVERRIDES", default={})
    LLM_CONTEXT_SAFETY_MARGIN_TOKENS = _int_setting("LLM_CONTEXT_SAFETY_MARGIN_TOKENS", default=256)

    local_defaults = bool(LOCAL_BACKEND) or LLM_PROVIDER in {"local", "llamacpp", "ollama"}
    LLM_MAX_CONCURRENCY = _int_setting("LLM_MAX_CONCURRENCY", default=1 if local_defaults else 5)
    LLM_REQUESTS_PER_MINUTE = _int_setting("LLM_REQUESTS_PER_MINUTE", default=45 if local_defaults else 240)
    LLM_TOKENS_PER_MINUTE = _int_setting(
        "LLM_TOKENS_PER_MINUTE",
        default=120_000 if local_defaults else 2_000_000,
    )
    LLM_MAX_RETRIES = _int_setting("LLM_MAX_RETRIES", default=3)
    LLM_BACKOFF_BASE_SECONDS = _float_setting("LLM_BACKOFF_BASE_SECONDS", default=1.0)
    LLM_BACKOFF_MAX_SECONDS = _float_setting("LLM_BACKOFF_MAX_SECONDS", default=30.0)
    LLM_BACKOFF_JITTER = _float_setting("LLM_BACKOFF_JITTER", default=0.5)
    LLM_TIMEOUT_SECONDS = _int_setting("LLM_TIMEOUT_SECONDS", default=180)
    LLM_TIMEOUT_SECONDS_LOCAL = _int_setting(
        "LLM_TIMEOUT_SECONDS_LOCAL",
        default=600 if local_defaults else None,
    )
    OLLAMA_SHOW_URL = _string_setting("OLLAMA_SHOW_URL")

    _CONFIG_METADATA = _CONFIG_METADATA
    _CONFIG_REPORT_EMITTED = False

    @classmethod
    def _format_entry_note(cls, name, metadata):
        source = metadata.get("source")
        reason = metadata.get("reason")
        default_value = metadata.get("default_value", _MISSING)
        raw_value = metadata.get("raw_value", _MISSING)
        if source == "default":
            if reason == "blank":
                return f"custom value was blank; using default {_format_config_value(name, default_value)}."
            return f"no custom value provided; using default {_format_config_value(name, default_value)}."
        if source == "fallback_default":
            if reason == "null":
                return f"custom value is null; using default {_format_config_value(name, default_value)}."
            return (
                f"custom value {_format_config_value(name, raw_value)} is invalid; "
                f"using default {_format_config_value(name, default_value)}."
            )
        if source == "unset":
            return "no custom value provided and no default is configured."
        return "using custom value from env_vars.json."

    @classmethod
    def get_effective_config_entries(cls, redact=True):
        entries = []
        for name in sorted(cls._CONFIG_METADATA):
            metadata = dict(cls._CONFIG_METADATA[name])
            value = getattr(cls, name, metadata.get("value"))
            entries.append(
                {
                    "key": name,
                    "env_key": metadata.get("env_key", name),
                    "value": _format_config_value(name, value, redact=redact),
                    "source": metadata.get("source", "unknown"),
                    "note": cls._format_entry_note(name, metadata),
                }
            )
        return entries

    @classmethod
    def emit_config_report(cls, printer=print, force=False):
        if cls._CONFIG_REPORT_EMITTED and not force:
            return

        entries = cls.get_effective_config_entries(redact=True)
        printer("Effective config:")
        for entry in entries:
            label = entry["key"]
            if entry["env_key"] != entry["key"]:
                label = f"{label} ({entry['env_key']})"
            printer(f"  {label} = {entry['value']} [{entry['source']}]")

        for entry in entries:
            label = entry["key"]
            if entry["env_key"] != entry["key"]:
                label = f"{label} ({entry['env_key']})"
            if entry["source"] == "default":
                printer(f"Config default used for {label}: {entry['note']}")
            elif entry["source"] == "fallback_default":
                printer(f"Config fallback default used for {label}: {entry['note']}")
            elif entry["source"] == "unset":
                printer(f"Config value unset for {label}: {entry['note']}")

        cls._CONFIG_REPORT_EMITTED = True

    @classmethod
    def validate_config(cls):
        cls.emit_config_report()

        def _is_missing(value):
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            return False

        errors = []
        required_keys = ["LLM_PROVIDER", "SECRET_KEY", "USER_EMAIL", "SEMANTIC_SCHOLAR_API_KEY"]
        if cls.LLM_PROVIDER == "azure-openai":
            required_keys.extend(["LLM_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"])
        elif cls.LLM_PROVIDER == "openai":
            required_keys.append("LLM_API_KEY")
        elif cls.LLM_PROVIDER == "openrouter":
            required_keys.append("LLM_API_KEY")
        elif cls.LLM_PROVIDER == "llamacpp" or cls.LLM_PROVIDER == "local":
            required_keys.append("LLM_BASE_URL")
        elif cls.LLM_PROVIDER == "azure-inference":
            required_keys.extend(["AZURE_AI_INFERENCE_ENDPOINT", "LLM_API_KEY"])

        if cls.ENABLE_EMAIL_NOTIFICATIONS:
            required_keys.extend(["EMAIL_SENDER", "EMAIL_APP_PASSWORD", "SMTP_SERVER", "SMTP_PORT", "BASE_URL"])

        if cls.REQUIRE_PASSWORD and not cls.ACCESS_PASSWORD:
            required_keys.append("ACCESS_PASSWORD")

        missing_keys = [key for key in required_keys if _is_missing(getattr(cls, key, None))]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        numeric_rules = [
            ("LLM_MAX_CONCURRENCY", cls.LLM_MAX_CONCURRENCY, 1, None),
            ("LLM_REQUESTS_PER_MINUTE", cls.LLM_REQUESTS_PER_MINUTE, 1, None),
            ("LLM_TOKENS_PER_MINUTE", cls.LLM_TOKENS_PER_MINUTE, 1, None),
            ("LLM_MAX_RETRIES", cls.LLM_MAX_RETRIES, 0, None),
            ("LLM_TIMEOUT_SECONDS", cls.LLM_TIMEOUT_SECONDS, 5, 7200),
            ("LLM_CONTEXT_SAFETY_MARGIN_TOKENS", cls.LLM_CONTEXT_SAFETY_MARGIN_TOKENS, 0, None),
            ("TRACE_EMBED_MAX_BYTES", cls.TRACE_EMBED_MAX_BYTES, 1024, None),
            ("TRACE_STACKTRACE_MAX_BYTES", cls.TRACE_STACKTRACE_MAX_BYTES, 256, None),
        ]
        for name, value, min_value, max_value in numeric_rules:
            if value is None:
                errors.append(f"{name} must be set.")
                continue
            try:
                numeric_value = int(value)
            except Exception:
                errors.append(f"{name} must be an integer.")
                continue
            if numeric_value < min_value:
                errors.append(f"{name} must be >= {min_value}.")
            if max_value is not None and numeric_value > max_value:
                errors.append(f"{name} must be <= {max_value}.")

        if cls.LLM_TIMEOUT_SECONDS_LOCAL is not None:
            try:
                timeout_local = int(cls.LLM_TIMEOUT_SECONDS_LOCAL)
                if timeout_local < 5 or timeout_local > 7200:
                    errors.append("LLM_TIMEOUT_SECONDS_LOCAL must be between 5 and 7200 seconds.")
            except Exception:
                errors.append("LLM_TIMEOUT_SECONDS_LOCAL must be an integer when provided.")

        try:
            backoff_base = float(cls.LLM_BACKOFF_BASE_SECONDS)
            backoff_max = float(cls.LLM_BACKOFF_MAX_SECONDS)
            backoff_jitter = float(cls.LLM_BACKOFF_JITTER)
            if backoff_base <= 0:
                errors.append("LLM_BACKOFF_BASE_SECONDS must be > 0.")
            if backoff_max <= 0:
                errors.append("LLM_BACKOFF_MAX_SECONDS must be > 0.")
            if backoff_max < backoff_base:
                errors.append("LLM_BACKOFF_MAX_SECONDS must be >= LLM_BACKOFF_BASE_SECONDS.")
            if backoff_jitter < 0 or backoff_jitter > 1:
                errors.append("LLM_BACKOFF_JITTER must be between 0 and 1.")
        except Exception:
            errors.append("LLM backoff settings must be numeric.")

        routing = cls.LLM_ROUTING
        if routing and not isinstance(routing, dict):
            errors.append("LLM_ROUTING must be a JSON object/dict.")
        elif isinstance(routing, dict):
            tasks = routing.get("tasks", {})
            if tasks is not None and not isinstance(tasks, dict):
                errors.append("LLM_ROUTING.tasks must be a dict when provided.")
            elif isinstance(tasks, dict):
                for task_name, task_cfg in tasks.items():
                    if not isinstance(task_cfg, dict):
                        errors.append(f"LLM_ROUTING.tasks.{task_name} must be a dict.")
                        continue
                    max_output = task_cfg.get("max_output_tokens")
                    if max_output is not None:
                        try:
                            max_output_i = int(max_output)
                            if max_output_i <= 0:
                                errors.append(f"LLM_ROUTING.tasks.{task_name}.max_output_tokens must be > 0.")
                        except Exception:
                            errors.append(
                                f"LLM_ROUTING.tasks.{task_name}.max_output_tokens must be an integer."
                            )
                    timeout_seconds = task_cfg.get("timeout_seconds")
                    if timeout_seconds is not None:
                        try:
                            timeout_i = int(timeout_seconds)
                            if timeout_i < 5 or timeout_i > 7200:
                                errors.append(
                                    f"LLM_ROUTING.tasks.{task_name}.timeout_seconds must be between 5 and 7200."
                                )
                        except Exception:
                            errors.append(
                                f"LLM_ROUTING.tasks.{task_name}.timeout_seconds must be an integer."
                            )

        if errors:
            raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
