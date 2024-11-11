import openai
import json
from app.config.settings import Config
from typing import Any, Optional, List
from openai import AzureOpenAI, AsyncAzureOpenAI
import asyncio

class OpenAIService:
    def __init__(self):
        if Config.USE_AZURE_OPENAI:
            self.client = AzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
            self.async_client = AsyncAzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION
            )
        else:
            self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            self.async_client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o") -> str:
        # Check that the prompt is not too long - if it is, return a failure message
        if len(prompt) + len(system_prompt or "") > 320000:
            return "Error: Prompt is too long"

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        self._update_token_usage(response.usage)
        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o") -> Any:
        # Check that the prompt is not too long - if it is, return a failure message
        if len(prompt) + len(system_prompt or "") > 320000:
            return json.loads('{"error": "Prompt is too long"}')

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
        
        self._update_token_usage(response.usage)
        return json.loads(response.choices[0].message.content)

    def _update_token_usage(self, usage):
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_cost += self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        # Prices as of May 2023 for GPT-4 (adjust as needed)
        prompt_price_per_1k = 0.0025
        completion_price_per_1k = 0.01

        prompt_cost = (prompt_tokens / 1000) * prompt_price_per_1k
        completion_cost = (completion_tokens / 1000) * completion_price_per_1k

        return prompt_cost + completion_cost

    def get_usage_stats(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": round(self.total_cost, 4)
        }

    def evaluate_abstract_confidence(self, text: str) -> float:
        """
        Evaluate the confidence that the given text is an abstract.
        Returns a confidence score between 0 and 1.
        """
        system_prompt = "You are an expert in identifying scientific abstracts. Given the following text, provide a confidence score between 0 and 1 indicating how likely it is to be an abstract."
        user_message = f"Text: {text}\n\nProvide a confidence score."

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        return float(response.choices[0].message.content.strip())

    async def process_claims_batch(self, claims: List[str], system_prompt: Optional[str] = None) -> List[dict]:
        """Process a batch of claims asynchronously"""
        tasks = []
        for claim in claims:
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant. Please validate the claim format and suggest improvements."},
                {"role": "user", "content": claim}
            ]
            
            task = self.async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            
            for response, original_claim in zip(responses, claims):
                if isinstance(response, Exception):
                    print(f"Error processing claim: {str(response)}")
                    results.append({
                        'original': original_claim,
                        'enhanced': original_claim,
                        'is_valid': False,
                        'explanation': f"Error: {str(response)}"
                    })
                    continue
                    
                self._update_token_usage(response.usage)
                results.append({
                    'original': original_claim,
                    'enhanced': response.choices[0].message.content,
                    'is_valid': True,
                    'explanation': ''
                })
            
            return results
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            raise
