import json
from pathlib import Path
import os

# Get the path to the env_vars.json file
env_file_path = Path(__file__).parent.parent / 'config/env_vars.json'

# Load the JSON file
try:
    with open(env_file_path, 'r') as f:
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


def _as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _as_dict(value, default=None):
    if default is None:
        default = {}
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return default
    return default

class Config:
    SECRET_KEY = env_vars.get('FLASK_SECRET_KEY')
    SEMANTIC_SCHOLAR_API_KEY = env_vars.get('SEMANTIC_SCHOLAR_API_KEY')
    USER_EMAIL = env_vars.get('USER_EMAIL')

    # Azure OpenAI configuration
    AZURE_OPENAI_ENDPOINT = env_vars.get('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_VERSION = env_vars.get('AZURE_OPENAI_API_VERSION', '2024-06-01')

    # Azure AI Inference configuration (for Phi-4 and similar models)
    AZURE_AI_INFERENCE_ENDPOINT = env_vars.get('AZURE_AI_INFERENCE_ENDPOINT')

    # Email notification configuration
    ENABLE_EMAIL_NOTIFICATIONS = _as_bool(env_vars.get('ENABLE_EMAIL_NOTIFICATIONS', False), default=False)
    EMAIL_SENDER = env_vars.get('EMAIL_SENDER')  # Gmail address
    EMAIL_APP_PASSWORD = env_vars.get('EMAIL_APP_PASSWORD')  # Gmail app password
    SMTP_SERVER = env_vars.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = _as_int(env_vars.get('SMTP_PORT', 587), default=587)
    BASE_URL = env_vars.get('BASE_URL', 'NO URL SET')

    # Add password configuration
    REQUIRE_PASSWORD = _as_bool(env_vars.get('REQUIRE_PASSWORD', False), default=False)
    ACCESS_PASSWORD = env_vars.get('ACCESS_PASSWORD')

    # AI Service Settings
    LLM_PROVIDER = env_vars.get("LLM_PROVIDER", "openai")  # 'azure-openai', 'azure-inference', 'openai', 'openrouter', or 'local'
    LLM_BASE_URL = env_vars.get("LLM_BASE_URL", "http://localhost:8000")
    LLM_API_KEY = env_vars.get("LLM_API_KEY", "")
    LLM_EVALUATION_MODEL = env_vars.get("LLM_EVALUATION_MODEL", "gpt-4o")
    LLM_HTTP_REFERER = env_vars.get("LLM_HTTP_REFERER")
    LLM_SITE_NAME = env_vars.get("LLM_SITE_NAME")
    LOCAL_BACKEND = env_vars.get("LOCAL_BACKEND", "")
    LOCAL_MODEL_CONTEXT_OVERRIDE = _as_int(env_vars.get("LOCAL_MODEL_CONTEXT_OVERRIDE", 0), default=0) or None

    # Rate limiting configuration (optional)
    RATE_LIMIT_MAX_TOKENS_PER_CLAIM = _as_int(env_vars.get("RATE_LIMIT_MAX_TOKENS_PER_CLAIM", 300000), default=300000)
    RATE_LIMIT_MAX_TOKENS_PER_WINDOW = _as_int(env_vars.get("RATE_LIMIT_MAX_TOKENS_PER_WINDOW", 25000), default=25000)
    RATE_LIMIT_MAX_REQUESTS_PER_WINDOW = _as_int(env_vars.get("RATE_LIMIT_MAX_REQUESTS_PER_WINDOW", 5), default=5)
    RATE_LIMIT_WINDOW_SIZE_SECONDS = _as_int(env_vars.get("RATE_LIMIT_WINDOW_SIZE_SECONDS", 10), default=10)

    # LLM gateway controls
    TRACE_ENABLED = _as_bool(env_vars.get("TRACE_ENABLED", True), default=True)
    TRACE_DIR = env_vars.get("TRACE_DIR", "saved_jobs")
    TRACE_EMBED_MODE = env_vars.get("TRACE_EMBED_MODE", "capped")
    TRACE_EMBED_MAX_BYTES = _as_int(env_vars.get("TRACE_EMBED_MAX_BYTES", 2_000_000), default=2_000_000)
    TRACE_STACKTRACE_MAX_BYTES = _as_int(env_vars.get("TRACE_STACKTRACE_MAX_BYTES", 4_000), default=4_000)
    TRACE_ALWAYS_WRITE_FILES = _as_bool(env_vars.get("TRACE_ALWAYS_WRITE_FILES", True), default=True)
    TRACE_COMPRESS_ON_COMPLETE = _as_bool(env_vars.get("TRACE_COMPRESS_ON_COMPLETE", False), default=False)

    LLM_ROUTING = _as_dict(env_vars.get("LLM_ROUTING"), default={})
    MODEL_REGISTRY_OVERRIDES = _as_dict(env_vars.get("MODEL_REGISTRY_OVERRIDES"), default={})
    LLM_CONTEXT_SAFETY_MARGIN_TOKENS = _as_int(env_vars.get("LLM_CONTEXT_SAFETY_MARGIN_TOKENS", 256), default=256)

    local_defaults = bool(LOCAL_BACKEND) or LLM_PROVIDER in {"local", "llamacpp", "ollama"}
    LLM_MAX_CONCURRENCY = _as_int(env_vars.get("LLM_MAX_CONCURRENCY", 1 if local_defaults else 5), default=1 if local_defaults else 5)
    LLM_REQUESTS_PER_MINUTE = _as_int(env_vars.get("LLM_REQUESTS_PER_MINUTE", 45 if local_defaults else 240), default=45 if local_defaults else 240)
    LLM_TOKENS_PER_MINUTE = _as_int(env_vars.get("LLM_TOKENS_PER_MINUTE", 120_000 if local_defaults else 2_000_000), default=120_000 if local_defaults else 2_000_000)
    LLM_MAX_RETRIES = _as_int(env_vars.get("LLM_MAX_RETRIES", 3), default=3)
    LLM_BACKOFF_BASE_SECONDS = _as_float(env_vars.get("LLM_BACKOFF_BASE_SECONDS", 1.0), default=1.0)
    LLM_BACKOFF_MAX_SECONDS = _as_float(env_vars.get("LLM_BACKOFF_MAX_SECONDS", 30.0), default=30.0)
    LLM_BACKOFF_JITTER = _as_float(env_vars.get("LLM_BACKOFF_JITTER", 0.5), default=0.5)
    LLM_TIMEOUT_SECONDS = _as_int(env_vars.get("LLM_TIMEOUT_SECONDS", 180), default=180)
    LLM_TIMEOUT_SECONDS_LOCAL = _as_int(
        env_vars.get("LLM_TIMEOUT_SECONDS_LOCAL", 600 if local_defaults else None),
        default=None,
    )
    OLLAMA_SHOW_URL = env_vars.get("OLLAMA_SHOW_URL")

    @classmethod
    def validate_config(cls):
        def _is_missing(value):
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            return False

        errors = []
        required_keys = ['LLM_PROVIDER', 'SECRET_KEY', 'USER_EMAIL', 'SEMANTIC_SCHOLAR_API_KEY']
        if cls.LLM_PROVIDER == "azure-openai":
            required_keys.extend(['LLM_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_VERSION'])
        elif cls.LLM_PROVIDER == "openai":
            required_keys.append('LLM_API_KEY')
        elif cls.LLM_PROVIDER == "openrouter":
            required_keys.append('LLM_API_KEY')
        elif cls.LLM_PROVIDER == "llamacpp" or cls.LLM_PROVIDER == "local":
            required_keys.append('LLM_BASE_URL')
        elif cls.LLM_PROVIDER == "azure-inference":
            required_keys.extend(['AZURE_AI_INFERENCE_ENDPOINT', 'LLM_API_KEY'])
        
        if cls.ENABLE_EMAIL_NOTIFICATIONS:
            required_keys.extend(['EMAIL_SENDER', 'EMAIL_APP_PASSWORD', 'SMTP_SERVER', 'SMTP_PORT', 'BASE_URL'])

        # Add password validation
        if cls.REQUIRE_PASSWORD and not cls.ACCESS_PASSWORD:
            required_keys.append('ACCESS_PASSWORD')

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
                            errors.append(f"LLM_ROUTING.tasks.{task_name}.max_output_tokens must be an integer.")
                    timeout_seconds = task_cfg.get("timeout_seconds")
                    if timeout_seconds is not None:
                        try:
                            timeout_i = int(timeout_seconds)
                            if timeout_i < 5 or timeout_i > 7200:
                                errors.append(
                                    f"LLM_ROUTING.tasks.{task_name}.timeout_seconds must be between 5 and 7200."
                                )
                        except Exception:
                            errors.append(f"LLM_ROUTING.tasks.{task_name}.timeout_seconds must be an integer.")

        if errors:
            raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
