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
    ENABLE_EMAIL_NOTIFICATIONS = env_vars.get('ENABLE_EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
    EMAIL_SENDER = env_vars.get('EMAIL_SENDER')  # Gmail address
    EMAIL_APP_PASSWORD = env_vars.get('EMAIL_APP_PASSWORD')  # Gmail app password
    SMTP_SERVER = env_vars.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(env_vars.get('SMTP_PORT', '587'))
    BASE_URL = env_vars.get('BASE_URL', 'NO URL SET')

    # Add password configuration
    REQUIRE_PASSWORD = env_vars.get('REQUIRE_PASSWORD', 'false').lower() == 'true'
    ACCESS_PASSWORD = env_vars.get('ACCESS_PASSWORD')

    # AI Service Settings
    LLM_PROVIDER = env_vars.get("LLM_PROVIDER", "openai")  # 'azure-openai', 'azure-inference', 'openai', or 'local'
    LLM_BASE_URL = env_vars.get("LLM_BASE_URL", "http://localhost:8000")
    LLM_API_KEY = env_vars.get("LLM_API_KEY", "")
    LLM_EVALUATION_MODEL = env_vars.get("LLM_EVALUATION_MODEL", "gpt-4o")


    @classmethod
    def validate_config(cls):
        required_keys = ['LLM_PROVIDER', 'SECRET_KEY', 'USER_EMAIL', 'SEMANTIC_SCHOLAR_API_KEY']
        if cls.LLM_PROVIDER == "azure-openai":
            required_keys.extend(['LLM_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_VERSION'])
        elif cls.LLM_PROVIDER == "openai":
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
