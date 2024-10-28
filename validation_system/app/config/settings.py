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
    OPENAI_API_KEY = env_vars.get('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = env_vars.get('ANTHROPIC_API_KEY')
    USER_EMAIL = env_vars.get('USER_EMAIL')
    FETCH_ABSTRACTS = True  # Set this to False if you don't want to fetch abstracts

    # Azure OpenAI configuration
    AZURE_OPENAI_API_KEY = env_vars.get('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT = env_vars.get('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_VERSION = env_vars.get('AZURE_OPENAI_API_VERSION', '2024-06-01')
    USE_AZURE_OPENAI = env_vars.get('USE_AZURE_OPENAI', 'false').lower() == 'true'

    @classmethod
    def validate_config(cls):
        required_keys = ['SECRET_KEY', 'SEMANTIC_SCHOLAR_API_KEY', 'USER_EMAIL']
        if cls.USE_AZURE_OPENAI:
            required_keys.extend(['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT'])
        else:
            required_keys.append('OPENAI_API_KEY')

        missing_keys = [key for key in required_keys if getattr(cls, key) is None]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
