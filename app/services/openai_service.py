import openai
import json
from app.config.settings import Config
from typing import Any, Optional
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.model = Config.LLM_EVALUATION_MODEL
        self.provider = Config.LLM_PROVIDER.lower()
        if self.provider == "azure":
            print(f"Using Azure OpenAI with model {self.model}")
            self.client = openai.AzureOpenAI(
                api_key=Config.LLM_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
            self.async_client = openai.AsyncAzureOpenAI(
                api_key=Config.LLM_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
        else:
            self.base_url = Config.LLM_BASE_URL
            if self.provider == "openai":
                print(f"Using OpenAI with model {self.model}")
                self.client = openai.OpenAI(api_key=Config.LLM_API_KEY)
                self.async_client = openai.AsyncOpenAI(api_key=Config.LLM_API_KEY)
            elif self.provider == "llamacpp":
                print(f"Using LlamaCPP with model {self.model}")
                self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key="sk-no-key-required")
                self.async_client = openai.AsyncOpenAI(base_url=self.base_url, api_key="sk-no-key-required")
        

    async def generate_json_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = None) -> Any:
        if len(prompt) + len(system_prompt or "") > 320000:
            return json.loads('{"error": "Prompt is too long"}')

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant. Please provide your response in valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        # Add jitter between 0 and 0.25 seconds
        jitter = random.uniform(0, 0.25)
        await asyncio.sleep(jitter)

        response = await self.async_client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        logger.info(f"API call completed for model {model or self.model}")
        return json.loads(response.choices[0].message.content)

    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = None) -> str:
        if len(prompt) + len(system_prompt or "") > 320000:
            return "Error: Prompt is too long"

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Add jitter between 0 and 0.25 seconds
        jitter = random.uniform(0, 0.25)
        await asyncio.sleep(jitter)

        response = await self.async_client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=0.0
        )
        
        logger.info(f"API call completed for model {model or self.model}")
        return response.choices[0].message.content
