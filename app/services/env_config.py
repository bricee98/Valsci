import errno
import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional

from app.config import settings as settings_module
from app.config.settings import Config


ENV_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WRITE_LOCK = Lock()

# Ordered categories for the Settings UI. Each entry: (id, label, description).
# Keys not listed in _SETTING_CATALOG fall into "advanced".
SETTING_CATEGORIES: List[Dict[str, str]] = [
    {"id": "provider", "label": "Default AI Provider", "description": "Bootstraps the “Default Config Provider” that appears on the Providers page. To run several providers/models or compare them, manage them on the Providers page instead — these values only seed and back that one default."},
    {"id": "routing", "label": "Model Routing & Token Budgets", "description": "Per-task model selection and output-token budgets. Raise the output budget here if a model is truncated before it can answer."},
    {"id": "performance", "label": "Performance & Rate Limits", "description": "Concurrency, throughput, retries, and timeouts for LLM calls."},
    {"id": "data", "label": "Literature Data", "description": "Semantic Scholar access and the corpus manifest used for evidence lookup."},
    {"id": "diagnostics", "label": "Diagnostics & Tracing", "description": "What Valsci records about each LLM call for debugging."},
    {"id": "email", "label": "Email Notifications", "description": "Optional email alerts when batches finish."},
    {"id": "security", "label": "Security & Access", "description": "Password protection and session security."},
    {"id": "general", "label": "General", "description": "Identity and base URL."},
    {"id": "paths", "label": "Storage Paths", "description": "Where Valsci reads and writes data on disk."},
    {"id": "advanced", "label": "Advanced", "description": "Uncategorized or rarely-changed keys."},
]

# Curated, human-friendly metadata per env key. `restart` marks settings that the
# background processor reads at startup — changing them requires restarting the
# processor (and sometimes the web app) to take effect, because env edits apply to
# the running web process in memory but not to the separate processor process.
_SETTING_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- AI Provider (seeds the default provider; full management is on the Providers page) ---
    "LLM_PROVIDER": {"category": "provider", "label": "Default Provider Type", "description": "Backend for the Default Config Provider: openai, azure-openai, azure-inference, openrouter, ollama, or llamacpp. The Providers page can enable additional providers regardless of this value.", "restart": True},
    "LLM_BASE_URL": {"category": "provider", "label": "Default Base URL", "description": "API base URL for the default provider. For Ollama in Docker use http://host.docker.internal:11434/v1. Editable per provider on the Providers page.", "restart": True},
    "LLM_API_KEY": {"category": "provider", "label": "Default API Key", "description": "Credential for the default provider (local backends accept any placeholder). Each provider on the Providers page has its own key.", "restart": True},
    "LLM_EVALUATION_MODEL": {"category": "provider", "label": "Default Model", "description": "Model used when a run does not specify one. Per-provider model lists are managed on the Providers page.", "restart": True},
    "LLM_HTTP_REFERER": {"category": "provider", "label": "HTTP Referer", "description": "Optional Referer header for the default provider (used by OpenRouter for attribution).", "restart": True},
    "LLM_SITE_NAME": {"category": "provider", "label": "Site Name", "description": "Optional site-name header for the default provider (used by OpenRouter for attribution).", "restart": True},
    "OLLAMA_SHOW_URL": {"category": "provider", "label": "Ollama Show URL", "description": "Endpoint for Ollama model metadata, e.g. http://host.docker.internal:11434/api/show. Used to discover local model details.", "restart": True},
    "AZURE_OPENAI_ENDPOINT": {"category": "provider", "label": "Azure OpenAI Endpoint", "description": "Resource endpoint for the default provider when its type is azure-openai.", "restart": True},
    "AZURE_OPENAI_API_VERSION": {"category": "provider", "label": "Azure OpenAI API Version", "description": "API version string for Azure OpenAI.", "restart": True},
    "AZURE_AI_INFERENCE_ENDPOINT": {"category": "provider", "label": "Azure AI Inference Endpoint", "description": "Endpoint for the default provider when its type is azure-inference.", "restart": True},
    "LOCAL_MODEL_CONTEXT_OVERRIDE": {"category": "provider", "label": "Local Context Override (fallback)", "description": "Fallback context-window size (tokens) used only for local models that have no explicit entry. Lower priority than: (1) Ollama's auto-detected window from /api/show, and (2) a per-model context window set on the Providers page — both of those win over this. 0 = no fallback (use the model's reported window, else 8192). This is total prompt+output capacity, not the per-task output budget (set those under Model Routing & Token Budgets).", "restart": True},
    # --- Routing & budgets ---
    "LLM_ROUTING": {"category": "routing", "label": "Task Routing & Budgets (advanced)", "description": "Optional per-stage overrides. By default the output budget for every stage comes from a model's Max Output Tokens on the Providers page — set it there. Only use this to give one stage a different model or output budget than the others. The per-stage editor above writes into this. Applies even when routing is disabled.", "restart": True},
    "MODEL_REGISTRY_OVERRIDES": {"category": "routing", "label": "Model Registry Overrides (fallback)", "description": "Fallback per-model overrides (context window, max output tokens, pricing) used only for runs that have no provider from the Providers page. The Providers page per-model settings take precedence — normally edit those there instead of here.", "restart": True},
    "LLM_CONTEXT_SAFETY_MARGIN_TOKENS": {"category": "routing", "label": "Context Safety Margin", "description": "Tokens reserved as headroom so prompts never exactly fill the context window.", "restart": True},
    "LLM_STRICT_JSON_SCHEMA": {"category": "routing", "label": "Strict JSON Schema", "description": "Force structured-output stages (final report, query generation) to match an exact JSON schema instead of just 'valid JSON'. Prevents models from inventing their own field names. If a provider rejects schema mode, it falls back to plain JSON automatically. Recommended on.", "restart": False},
    "LLM_JSON_REPAIR_PASS": {"category": "routing", "label": "JSON Repair Pass", "description": "When a model returns valid JSON with the wrong fields, run a second LLM pass that reshapes its output into the required schema without changing the content, then re-validate. Costs an extra call only when the first output fails. Off by default.", "restart": False},
    # --- Performance ---
    "LLM_MAX_CONCURRENCY": {"category": "performance", "label": "Max Concurrency", "description": "Maximum simultaneous in-flight LLM requests.", "restart": True},
    "LLM_REQUESTS_PER_MINUTE": {"category": "performance", "label": "Requests / Minute", "description": "Request rate cap for the provider.", "restart": True},
    "LLM_TOKENS_PER_MINUTE": {"category": "performance", "label": "Tokens / Minute", "description": "Token throughput cap for the provider.", "restart": True},
    "LLM_MAX_RETRIES": {"category": "performance", "label": "Max Retries", "description": "Retry attempts for transient LLM errors (rate limits, timeouts).", "restart": True},
    "LLM_BACKOFF_BASE_SECONDS": {"category": "performance", "label": "Backoff Base (s)", "description": "Initial wait before the first retry.", "restart": True},
    "LLM_BACKOFF_MAX_SECONDS": {"category": "performance", "label": "Backoff Max (s)", "description": "Ceiling on retry backoff wait.", "restart": True},
    "LLM_BACKOFF_JITTER": {"category": "performance", "label": "Backoff Jitter", "description": "Randomization factor (0–1) applied to backoff waits.", "restart": True},
    "LLM_TIMEOUT_SECONDS": {"category": "performance", "label": "Request Timeout (s)", "description": "Per-request timeout for hosted providers.", "restart": True},
    "LLM_TIMEOUT_SECONDS_LOCAL": {"category": "performance", "label": "Local Request Timeout (s)", "description": "Per-request timeout for local providers (often slower).", "restart": True},
    "RATE_LIMIT_MAX_TOKENS_PER_CLAIM": {"category": "performance", "label": "Max Tokens / Claim", "description": "Safety cap on total tokens spent evaluating a single claim.", "restart": True},
    "RATE_LIMIT_MAX_TOKENS_PER_WINDOW": {"category": "performance", "label": "Max Tokens / Window", "description": "Token budget per rate-limit window.", "restart": True},
    "RATE_LIMIT_MAX_REQUESTS_PER_WINDOW": {"category": "performance", "label": "Max Requests / Window", "description": "Request budget per rate-limit window.", "restart": True},
    "RATE_LIMIT_WINDOW_SIZE_SECONDS": {"category": "performance", "label": "Window Size (s)", "description": "Length of the rate-limit window.", "restart": True},
    # --- Data ---
    "SEMANTIC_SCHOLAR_API_KEY": {"category": "data", "label": "Semantic Scholar API Key", "description": "Required for downloading datasets and live paper lookup.", "restart": True},
    "SEMANTIC_SCHOLAR_MANIFEST": {"category": "data", "label": "Corpus Manifest", "description": "Filename (in semantic_scholar/manifests/) of the curated corpus to use.", "restart": True},
    "FETCH_REMOTE_CONTENT_ON_MISS": {"category": "data", "label": "Fetch Missing Content From Web", "description": "When a paper or author is not in the local corpus, fetch it from the live Semantic Scholar API (uses the API key, rate-limited): a paper's abstract/TLDR, and author h-index/citation counts for bibliometrics. Needed when live search returns papers outside a mini corpus. Toggle this from the Data page.", "restart": False},
    # --- Diagnostics ---
    "TRACE_ENABLED": {"category": "diagnostics", "label": "Enable Tracing", "description": "Record each LLM call (prompt, response, usage) for inspection.", "restart": True},
    "TRACE_EMBED_MODE": {"category": "diagnostics", "label": "Embed Mode", "description": "How much raw payload to embed in traces: full, capped, or none.", "restart": True},
    "TRACE_EMBED_MAX_BYTES": {"category": "diagnostics", "label": "Embed Max Bytes", "description": "Cap on embedded payload size per trace (capped mode).", "restart": True},
    "TRACE_STACKTRACE_MAX_BYTES": {"category": "diagnostics", "label": "Stacktrace Max Bytes", "description": "Cap on stored stacktrace length.", "restart": True},
    "TRACE_ALWAYS_WRITE_FILES": {"category": "diagnostics", "label": "Always Write Trace Files", "description": "Persist trace files even when embedding is disabled.", "restart": True},
    "TRACE_COMPRESS_ON_COMPLETE": {"category": "diagnostics", "label": "Compress On Complete", "description": "Gzip trace files once a claim finishes.", "restart": True},
    # --- Email ---
    "ENABLE_EMAIL_NOTIFICATIONS": {"category": "email", "label": "Enable Notifications", "description": "Send an email when a batch completes.", "restart": True},
    "EMAIL_SENDER": {"category": "email", "label": "Sender Address", "description": "From address for notification emails.", "restart": True},
    "EMAIL_APP_PASSWORD": {"category": "email", "label": "App Password", "description": "SMTP app password for the sender account.", "restart": True},
    "SMTP_SERVER": {"category": "email", "label": "SMTP Server", "description": "Outgoing mail server hostname.", "restart": True},
    "SMTP_PORT": {"category": "email", "label": "SMTP Port", "description": "Outgoing mail server port (e.g. 587 for TLS).", "restart": True},
    # --- Security ---
    "REQUIRE_PASSWORD": {"category": "security", "label": "Require Password", "description": "Gate the whole app behind a shared access password.", "restart": True},
    "ACCESS_PASSWORD": {"category": "security", "label": "Access Password", "description": "Shared password when password protection is enabled.", "restart": True},
    "FLASK_SECRET_KEY": {"category": "security", "label": "Flask Secret Key", "description": "Signs session cookies. Use a long random value.", "restart": True},
    # --- General ---
    "USER_EMAIL": {"category": "general", "label": "User Email", "description": "Your email, used as a contact for Semantic Scholar and notifications."},
    "BASE_URL": {"category": "general", "label": "Base URL", "description": "Public URL of this Valsci instance, used in notification links."},
    # --- Paths ---
    "STATE_DIR": {"category": "paths", "label": "State Directory", "description": "Canonical claim store, runs, arenas, and job history.", "restart": True},
    "PROVIDER_CATALOG_PATH": {"category": "paths", "label": "Provider Catalog Path", "description": "Where provider/model definitions are stored.", "restart": True},
    "TRACE_DIR": {"category": "paths", "label": "Trace Directory", "description": "Where LLM trace and issue files are written.", "restart": True},
    "SAVED_JOBS_DIR": {"category": "paths", "label": "Saved Jobs Directory", "description": "Completed claim results.", "restart": True},
    "QUEUED_JOBS_DIR": {"category": "paths", "label": "Queued Jobs Directory", "description": "Claims waiting to be processed.", "restart": True},
    "MIGRATION_ARCHIVE_DIR": {"category": "paths", "label": "Migration Archive Directory", "description": "Where legacy batches are archived after migration.", "restart": True},
}

# Settings the LLM GatewayFactory reads fresh from Config when it builds a
# gateway (which it does per claim). Because the processor re-applies env_vars.json
# to Config when the file changes, edits to these take effect on the next claim
# with no restart. Everything else is captured at processor startup (rate limits,
# email, storage paths, …) and needs a full restart — so restart_required is the
# complement of this set.
_HOT_RELOAD_KEYS = {
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_EVALUATION_MODEL",
    "LLM_HTTP_REFERER", "LLM_SITE_NAME", "OLLAMA_SHOW_URL",
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_AI_INFERENCE_ENDPOINT",
    "LOCAL_MODEL_CONTEXT_OVERRIDE",
    "LLM_ROUTING", "MODEL_REGISTRY_OVERRIDES", "LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
    "LLM_STRICT_JSON_SCHEMA", "LLM_JSON_REPAIR_PASS",
    "LLM_MAX_CONCURRENCY", "LLM_REQUESTS_PER_MINUTE", "LLM_TOKENS_PER_MINUTE",
    "LLM_MAX_RETRIES", "LLM_BACKOFF_BASE_SECONDS", "LLM_BACKOFF_MAX_SECONDS",
    "LLM_BACKOFF_JITTER", "LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS_LOCAL",
    "TRACE_ENABLED", "TRACE_EMBED_MODE", "TRACE_EMBED_MAX_BYTES", "TRACE_STACKTRACE_MAX_BYTES",
    # Read at content-lookup time by the searcher, so a processor config reload
    # applies it without a restart.
    "FETCH_REMOTE_CONTENT_ON_MISS",
}


def _restart_required(env_key: str) -> bool:
    return env_key not in _HOT_RELOAD_KEYS


# The four pipeline stages whose per-task output-token budget is surfaced as a
# friendly editor on top of the raw LLM_ROUTING JSON.
ROUTING_TASK_STAGES: List[Dict[str, str]] = [
    {"key": "query_generation", "label": "Query Generation"},
    {"key": "paper_analysis", "label": "Paper Analysis"},
    {"key": "venue_scoring", "label": "Venue Scoring"},
    {"key": "final_report", "label": "Final Report"},
]


def _catalog_entry(env_key: str) -> Dict[str, Any]:
    info = _SETTING_CATALOG.get(env_key)
    if info:
        return info
    return {"category": "advanced", "label": env_key, "description": "", "restart": False}


def routing_output_budgets(routing_value: Any) -> Dict[str, Optional[int]]:
    """Extract per-stage max_output_tokens from an LLM_ROUTING object value."""
    budgets: Dict[str, Optional[int]] = {}
    tasks = routing_value.get("tasks", {}) if isinstance(routing_value, dict) else {}
    for stage in ROUTING_TASK_STAGES:
        task_cfg = tasks.get(stage["key"]) if isinstance(tasks, dict) else None
        value = task_cfg.get("max_output_tokens") if isinstance(task_cfg, dict) else None
        budgets[stage["key"]] = value if isinstance(value, int) else None
    return budgets
_PATH_CONFIG_KEYS = {
    "SAVED_JOBS_DIR",
    "QUEUED_JOBS_DIR",
    "STATE_DIR",
    "MIGRATION_ARCHIVE_DIR",
    "PROVIDER_CATALOG_PATH",
    "TRACE_DIR",
}
_SENSITIVE_ENV_KEYS = {
    "FLASK_SECRET_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "LLM_API_KEY",
    "EMAIL_APP_PASSWORD",
    "ACCESS_PASSWORD",
}


def env_vars_path() -> Path:
    return Path(settings_module.env_file_path)


def env_file_mtime() -> Optional[float]:
    """Modification time of env_vars.json, or None if it cannot be read."""
    try:
        return env_vars_path().stat().st_mtime
    except OSError:
        return None


def example_env_vars_path() -> Path:
    return settings_module.PROJECT_ROOT / "env_vars.json.example"


def _clone(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_env_vars() -> Dict[str, Any]:
    return _read_json_file(env_vars_path())


def _read_example_env_vars() -> Dict[str, Any]:
    path = example_env_vars_path()
    if not path.exists():
        return {}
    return _read_json_file(path)


def _metadata_by_env_key() -> Dict[str, Dict[str, Any]]:
    by_env_key: Dict[str, Dict[str, Any]] = {}
    for config_key, metadata in Config._CONFIG_METADATA.items():
        env_key = str(metadata.get("env_key") or config_key)
        by_env_key[env_key] = {"config_key": config_key, **metadata}
    return by_env_key


def _ordered_env_keys(raw: Mapping[str, Any], example: Mapping[str, Any], metadata: Mapping[str, Any]) -> List[str]:
    keys: List[str] = []
    for source in (raw, example, metadata):
        for key in source:
            if key not in keys:
                keys.append(key)
    return keys


def _is_sensitive(env_key: str, config_key: Optional[str]) -> bool:
    return (
        env_key in _SENSITIVE_ENV_KEYS
        or (config_key or "") in settings_module._SENSITIVE_KEYS
        or env_key in settings_module._SENSITIVE_KEYS
    )


def _infer_value_type(value: Any, fallback_value: Any = None) -> str:
    sample = fallback_value if fallback_value is not None else value
    if isinstance(sample, bool):
        return "boolean"
    if isinstance(sample, int) and not isinstance(sample, bool):
        return "integer"
    if isinstance(sample, float):
        return "number"
    if isinstance(sample, list):
        return "array"
    if isinstance(sample, dict):
        return "object"
    sample = value
    if isinstance(sample, bool):
        return "boolean"
    if isinstance(sample, int) and not isinstance(sample, bool):
        return "integer"
    if isinstance(sample, float):
        return "number"
    if isinstance(sample, list):
        return "array"
    if isinstance(sample, dict):
        return "object"
    return "string"


def _coerce_raw_editor_value(raw_value: Any, fallback_value: Any) -> Any:
    if isinstance(fallback_value, bool):
        return settings_module._as_bool(raw_value, default=fallback_value)
    if isinstance(fallback_value, int) and not isinstance(fallback_value, bool):
        if raw_value in (None, ""):
            return ""
        try:
            return int(raw_value)
        except Exception:
            return raw_value
    if isinstance(fallback_value, float):
        if raw_value in (None, ""):
            return ""
        try:
            return float(raw_value)
        except Exception:
            return raw_value
    if isinstance(fallback_value, dict):
        if isinstance(raw_value, dict):
            return dict(raw_value)
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return raw_value
    if isinstance(fallback_value, list):
        if isinstance(raw_value, list):
            return list(raw_value)
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        return raw_value
    return _clone(raw_value)


def _editor_value(raw_present: bool, raw_value: Any, metadata: Mapping[str, Any], example_value: Any) -> Any:
    fallback = metadata.get("default_value", metadata.get("value", example_value))
    if raw_present:
        return _coerce_raw_editor_value(raw_value, fallback)
    if "default_value" in metadata:
        return _clone(metadata.get("default_value"))
    if "value" in metadata:
        return _clone(metadata.get("value"))
    if example_value is not None:
        return _clone(example_value)
    return ""


def build_env_config_state() -> Dict[str, Any]:
    raw = read_env_vars()
    example = _read_example_env_vars()
    metadata_by_env = _metadata_by_env_key()
    effective_by_env = {
        entry["env_key"]: entry
        for entry in Config.get_effective_config_entries(redact=True)
    }
    entries = []
    for env_key in _ordered_env_keys(raw, example, metadata_by_env):
        metadata = metadata_by_env.get(env_key, {})
        effective = effective_by_env.get(env_key, {})
        config_key = metadata.get("config_key")
        raw_present = env_key in raw
        raw_value = raw.get(env_key)
        value = _editor_value(raw_present, raw_value, metadata, example.get(env_key))
        fallback = metadata.get("default_value", metadata.get("value", example.get(env_key)))
        catalog = _catalog_entry(env_key)
        entries.append(
            {
                "env_key": env_key,
                "config_key": config_key,
                "raw_present": raw_present,
                "value": value,
                "value_type": _infer_value_type(value, fallback),
                "sensitive": _is_sensitive(env_key, config_key),
                "effective_value": effective.get("value", "<untracked>"),
                "source": effective.get("source", "env_vars.json" if raw_present else "unset"),
                "note": effective.get("note", "Custom env_vars.json key." if raw_present else "Not configured."),
                "category": catalog.get("category", "advanced"),
                "label": catalog.get("label") or env_key,
                "description": catalog.get("description", ""),
                "restart_required": _restart_required(env_key),
            }
        )

    # Group entries by category, preserving the curated category order, and
    # surface the per-stage output-token budgets parsed from LLM_ROUTING so the UI
    # can offer a friendly editor for the most commonly-tuned values.
    entries_by_key = {entry["env_key"]: entry for entry in entries}
    groups = []
    seen_keys: set = set()
    for category in SETTING_CATEGORIES:
        group_entries = [entry for entry in entries if entry["category"] == category["id"]]
        for entry in group_entries:
            seen_keys.add(entry["env_key"])
        if group_entries:
            groups.append({**category, "entries": group_entries})

    routing_entry = entries_by_key.get("LLM_ROUTING")
    routing_budgets = routing_output_budgets(routing_entry["value"] if routing_entry else None)

    return {
        "path": str(env_vars_path()),
        "example_path": str(example_env_vars_path()),
        "entries": entries,
        "groups": groups,
        "categories": SETTING_CATEGORIES,
        "routing_task_stages": ROUTING_TASK_STAGES,
        "routing_output_budgets": routing_budgets,
    }


def _validate_updates(updates: Mapping[str, Any]) -> Dict[str, Any]:
    validated: Dict[str, Any] = {}
    for key, value in updates.items():
        normalized_key = str(key or "").strip()
        if not ENV_KEY_PATTERN.match(normalized_key):
            raise ValueError(f"Invalid env var key: {key}")
        if value is None:
            validated[normalized_key] = ""
        elif isinstance(value, (str, int, float, bool, dict, list)):
            validated[normalized_key] = value
        else:
            raise ValueError(f"{normalized_key} must be a JSON string, number, boolean, object, or array")
    return validated


def update_env_vars(updates: Mapping[str, Any]) -> Dict[str, Any]:
    validated = _validate_updates(updates)
    with _WRITE_LOCK:
        path = env_vars_path()
        raw = read_env_vars()
        raw.update(validated)
        content = json.dumps(raw, indent=2, ensure_ascii=True) + "\n"
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        try:
            temporary_path.replace(path)
        except OSError as exc:
            # A bind-mounted config file (common in Docker, where env_vars.json is
            # mounted as a single file) cannot be replaced by rename — the mount
            # point is busy / on another device. Fall back to an in-place write,
            # which keeps the mount intact and still updates the file's mtime.
            if exc.errno in (errno.EBUSY, errno.EXDEV, errno.EPERM):
                with path.open("w", encoding="utf-8") as handle:
                    handle.write(content)
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            else:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
                raise
    return raw


def _convert_runtime_value(config_key: str, raw_value: Any, metadata: Mapping[str, Any]) -> Any:
    default = metadata.get("default_value", metadata.get("value", getattr(Config, config_key, None)))
    current = getattr(Config, config_key, default)

    if config_key in _PATH_CONFIG_KEYS:
        candidate = raw_value if raw_value not in (None, "") else default
        return settings_module._resolve_project_path(candidate)
    if isinstance(default, bool) or isinstance(current, bool):
        return settings_module._as_bool(raw_value, default=bool(default))
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw_value)
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(raw_value)
        except Exception:
            return default
    if isinstance(default, dict) or isinstance(current, dict):
        if isinstance(raw_value, dict):
            return dict(raw_value)
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return dict(default) if isinstance(default, dict) else {}
    return raw_value


def apply_env_vars_to_runtime(raw: Optional[Mapping[str, Any]] = None, app_config: Optional[Dict[str, Any]] = None) -> None:
    raw_values = dict(raw if raw is not None else read_env_vars())
    settings_module.env_vars.clear()
    settings_module.env_vars.update(raw_values)

    for config_key, metadata in Config._CONFIG_METADATA.items():
        env_key = str(metadata.get("env_key") or config_key)
        if env_key not in raw_values:
            continue
        value = _convert_runtime_value(config_key, raw_values[env_key], metadata)
        if config_key == "LOCAL_MODEL_CONTEXT_OVERRIDE" and value == 0:
            value = None
        setattr(Config, config_key, value)
        metadata["value"] = _clone(value)
        metadata["raw_value"] = _clone(raw_values[env_key])
        metadata["source"] = "env_vars.json"
        metadata["reason"] = "provided"
        if app_config is not None:
            app_config[config_key] = value
