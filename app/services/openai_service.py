import openai
import json
from app.config.settings import Config
from typing import Any, Optional
import asyncio
import random
import logging
from asyncio import timeout
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import httpx

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.model = Config.LLM_EVALUATION_MODEL
        self.provider = Config.LLM_PROVIDER.lower()
        self.count = 0
        if self.provider == "azure-openai":
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
        elif self.provider == "azure-inference":
            print(f"Using Azure AI Inference with model {Config.LLM_EVALUATION_MODEL}")
            self.model = Config.LLM_EVALUATION_MODEL
            self.endpoint = Config.AZURE_AI_INFERENCE_ENDPOINT
            self.api_key = Config.LLM_API_KEY
            self.ai_chat_client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            # We'll use httpx for async calls
            self.async_client = None  # Not using OpenAI's async client for Azure AI Inference
        else:
            self.base_url = Config.LLM_BASE_URL
            if self.provider == "openai":
                print(f"Using OpenAI with model {self.model}")
                self.client = openai.OpenAI(api_key=Config.LLM_API_KEY)
                self.async_client = openai.AsyncOpenAI(api_key=Config.LLM_API_KEY)
            elif self.provider == "llamacpp" or self.provider == "local":
                print(f"Using Alternative Model with model {self.model}")
                self.client = openai.OpenAI(base_url=self.base_url, api_key="sk-no-key-required")
                self.async_client = openai.AsyncOpenAI(base_url=self.base_url, api_key="sk-no-key-required")
        

    async def generate_json_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = None) -> Any:
        if len(prompt) + len(system_prompt or "") > 320000:
            return json.loads('{"error": "Prompt is too long"}')

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant. Please provide your response in valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        # Add jitter between 0 and 1.5 seconds
        jitter = random.uniform(0, 1.5)
        await asyncio.sleep(jitter)

        self.count += 1
        print(f"OAI Service Count: {self.count}")

        if self.provider == "azure-inference":
            # Use Azure AI Inference SDK for Phi-4 and other Azure AI models
            try:
                from azure.ai.inference.models import SystemMessage, UserMessage
                
                async with asyncio.timeout(180):  # 180 seconds = 3 minutes
                    # Create proper message formats for Azure AI Inference
                    ai_messages = [
                        SystemMessage(content=system_prompt or "You are a helpful assistant. Please provide your response in valid JSON format."),
                        UserMessage(content=prompt)
                    ]
                    
                    # Use asyncio to run the synchronous client in a thread pool
                    response = await asyncio.to_thread(
                        self.ai_chat_client.complete,
                        messages=ai_messages,
                        max_tokens=2048,
                        temperature=0.0,
                        model=model or self.model
                    )
                    
                    # Parse the response content as JSON
                    response_content = response.choices[0].message.content
                    json_content = json.loads(response_content)
                    
                    # Estimate token usage - this is just an approximation since Azure AI Inference doesn't report tokens
                    input_tokens = len(prompt) // 4 + len(system_prompt or "") // 4
                    output_tokens = len(response_content) // 4
                    
                    return {
                        'content': json_content,
                        'usage': {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'cost': 0.0  # We don't have standard pricing for Azure AI Inference models yet
                        }
                    }
            except Exception as e:
                logger.error(f"Error with Azure AI Inference SDK: {str(e)}")
                return {
                    'content': {"error": f"Azure AI Inference error: {str(e)}"},
                    'usage': {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'cost': 0.0
                    }
                }
        else:
            # Determine the effective model (explicitly provided or the default one)
            effective_model = model or self.model

            # Build the request parameters. Some models (e.g. "o3") do not support an explicit
            # temperature value, so we only set it when the model allows overriding.
            request_kwargs = {
                "model": effective_model,
                "messages": messages,
                "response_format": {"type": "json_object"}
            }

            if effective_model != "o3" and effective_model != "gpt-5":
                # Most models support temperature control; keep it at 0.0 for deterministic output
                request_kwargs["temperature"] = 0.0

            async with asyncio.timeout(180):  # 180 seconds = 3 minutes
                response = await self.async_client.chat.completions.create(**request_kwargs)
            
            logger.info(f"API call completed for model {model or self.model}")
            
            # Calculate and return token usage and cost along with the response
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.tokens_to_cost(input_tokens, output_tokens, model or self.model)
            
            return {
                'content': json.loads(response.choices[0].message.content),
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost
                }
            }

    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = None) -> str:
        if len(prompt) + len(system_prompt or "") > 320000:
            return "Error: Prompt is too long"

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Add jitter between 0 and 1.5 seconds
        jitter = random.uniform(0, 1.5)
        await asyncio.sleep(jitter)
        
        if self.provider == "azure-inference":
            # Use Azure AI Inference SDK for models like Phi-4
            try:
                from azure.ai.inference.models import SystemMessage, UserMessage
                
                async with asyncio.timeout(180):  # 180 seconds = 3 minutes
                    # Create proper message formats for Azure AI Inference
                    ai_messages = [
                        SystemMessage(content=system_prompt or "You are a helpful assistant."),
                        UserMessage(content=prompt)
                    ]
                    
                    # Use asyncio to run the synchronous client in a thread pool
                    response = await asyncio.to_thread(
                        self.ai_chat_client.complete,
                        messages=ai_messages,
                        max_tokens=2048,
                        temperature=0.0,
                        model=model or self.model
                    )
                    
                    # Get the response content as text
                    response_content = response.choices[0].message.content
                    
                    # Estimate token usage - this is just an approximation since Azure AI Inference doesn't report tokens
                    input_tokens = len(prompt) // 4 + len(system_prompt or "") // 4
                    output_tokens = len(response_content) // 4
                    
                    return {
                        'content': response_content,
                        'usage': {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'cost': 0.0  # We don't have standard pricing for Azure AI Inference models yet
                        }
                    }
            except Exception as e:
                logger.error(f"Error with Azure AI Inference SDK: {str(e)}")
                return {
                    'content': f"Azure AI Inference error: {str(e)}",
                    'usage': {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'cost': 0.0
                    }
                }
        else:
            effective_model = model or self.model

            request_kwargs = {
                "model": effective_model,
                "messages": messages
            }

            if effective_model != "o3" and effective_model != "gpt-5":
                request_kwargs["temperature"] = 0.0

            async with asyncio.timeout(180):  # 180 seconds = 3 minutes
                response = await self.async_client.chat.completions.create(**request_kwargs)
            
            logger.info(f"API call completed for model {model or self.model}")

            # Calculate and return token usage and cost along with the response
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.tokens_to_cost(input_tokens, output_tokens, model or self.model)
            
            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost
                }
            }
    
    def tokens_to_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        if model is None:
            return 0.0
        if model == "gpt-4o" or model == "gpt-4o-2":
            return (input_tokens * 2.50/1000000) + (output_tokens * 10.00/1000000)
        elif model == "gpt-4o-mini":
            return (input_tokens * 0.15/1000000) + (output_tokens * 0.6/1000000)
        elif model.startswith("phi-"):
            return 0.0  # For now, we don't have pricing info for Phi models
        elif model == "o3":
            return (input_tokens * 2.00/1000000) + (output_tokens * 8.00/1000000)
        elif model == "gpt-5":
            return (input_tokens * 1.25/1000000) + (output_tokens * 10.00/1000000)
        else:
            return 0.0