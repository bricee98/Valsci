import openai
import json
from app.config.settings import Config
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        if Config.USE_AZURE_OPENAI:
            self.client = openai.AzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
            self.async_client = openai.AsyncAzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
        else:
            self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            self.async_client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    async def generate_json_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4") -> Any:
        if len(prompt) + len(system_prompt or "") > 320000:
            return json.loads('{"error": "Prompt is too long"}')

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant. Please provide your response in valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        logger.info(f"API call completed for model {model}")
        return json.loads(response.choices[0].message.content)

    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4") -> str:
        if len(prompt) + len(system_prompt or "") > 320000:
            return "Error: Prompt is too long"

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        
        logger.info(f"API call completed for model {model}")
        return response.choices[0].message.content
