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
    OLLAMA_SHOW_URL = env_vars.get("OLLAMA_SHOW_URL")

    @classmethod
    def validate_config(cls):
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

        missing_keys = [key for key in required_keys if getattr(cls, key) is None]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
