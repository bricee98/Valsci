import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional

from app.config import settings as settings_module
from app.config.settings import Config


ENV_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WRITE_LOCK = Lock()
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
            }
        )

    return {
        "path": str(env_vars_path()),
        "example_path": str(example_env_vars_path()),
        "entries": entries,
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
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        temporary_path.replace(path)
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
