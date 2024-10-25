import openai
import json
from app.config.settings import Config
from typing import Any, Optional

class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o") -> str:
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o") -> Any:
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant. Please provide your response in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        return json.loads(response.choices[0].message.content)
