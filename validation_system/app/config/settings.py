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

    @classmethod
    def validate_config(cls):
        missing_keys = [key for key, value in vars(cls).items() if not key.startswith('__') and value is None]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
