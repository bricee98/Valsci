import openai
import json
from app.config.settings import Config
from typing import Any, Optional, List, Dict
from openai import AzureOpenAI, AsyncAzureOpenAI
import asyncio
import logging
from asyncio import TimeoutError
import time

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self, loop=None):
        self._loop = loop
        
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

    async def _make_request_with_timeout(self, **kwargs) -> Any:
        """Make a request with timeout and retry logic"""
        start_time = time.time()
        loop = self._loop or asyncio.get_running_loop()
        
        async def single_request():
            try:
                return await self.async_client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                raise

        async def timeout_wrapper(coro, timeout):
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except TimeoutError:
                logger.warning(f"Request timed out after {timeout} seconds")
                raise

        # Create initial task
        tasks = [
            loop.create_task(timeout_wrapper(single_request(), timeout=30))
        ]
        
        while True:
            try:
                # Wait for the first successful response or all failures
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                # Check results
                for task in done:
                    try:
                        result = await task
                        logger.info(f"Request completed in {time.time() - start_time:.2f} seconds")
                        return result
                    except Exception as e:
                        logger.warning(f"Task failed: {str(e)}")
                
                # If we get here, all tasks failed
                if len(tasks) >= 3:  # Max 3 concurrent requests
                    raise Exception("All requests failed after 3 attempts")
                
                # Add a new request to race with existing ones
                logger.info("Starting additional request due to timeout")
                tasks = [t for t in tasks if not t.done()]
                tasks.append(loop.create_task(timeout_wrapper(single_request(), timeout=30)))
                
            except Exception as e:
                if len(tasks) >= 3:  # Max 3 concurrent requests
                    raise Exception(f"All requests failed after 3 attempts: {str(e)}")
                
                # Start a new request
                logger.info("Starting additional request due to error")
                tasks = [t for t in tasks if not t.done()]
                tasks.append(loop.create_task(timeout_wrapper(single_request(), timeout=30)))

    async def _make_request(self, **kwargs):
        """Wrapper for the request with timeout logic"""
        loop = self._loop or asyncio.get_running_loop()
        return await loop.create_task(
            self._make_request_with_timeout(**kwargs)
        )

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
        logger.info(f"API call completed for model {model}")
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
        logger.info(f"API call completed for model {model}")
        return json.loads(response.choices[0].message.content)

    def _update_token_usage(self, usage):
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_cost += self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
        
        # Add logging for token usage
        logger.info(f"API call usage - Prompt tokens: {usage.prompt_tokens}, "
                   f"Completion tokens: {usage.completion_tokens}, "
                   f"Total tokens: {usage.prompt_tokens + usage.completion_tokens}")

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

    async def enhance_claims_batch(self, claims: List[str], system_prompt: Optional[str] = None) -> List[dict]:
        """Process a batch of claims asynchronously"""
        loop = self._loop or asyncio.get_running_loop()
        tasks = []
        
        # Use the same system prompt as in ClaimProcessor
        default_system_prompt = """
        You are an AI assistant tasked with evaluating scientific claims and optimizing them for search. 
        Respond with a JSON object containing 'is_valid' (boolean), 'explanation' (string), and 'suggested' (string).
        The 'is_valid' field should be true if the input is a proper scientific claim, and false otherwise. 
        The 'explanation' field should provide a brief reason for your decision.
        The 'suggested' field should always contain an optimized version of the claim, 
        even if the original claim is invalid. For invalid claims, provide a corrected or improved version.

        Examples of valid scientific claims (note: these may or may not be true, but they are properly formed claims):
         1. "Increased consumption of processed foods is linked to higher rates of obesity in urban populations."
         2. "The presence of certain gut bacteria can influence mood and cognitive function in humans."
         3. "Exposure to blue light from electronic devices before bedtime does not disrupt the circadian rhythm."
         4. "Regular meditation practice can lead to structural changes in the brain's gray matter."
         5. "Higher levels of atmospheric CO2 have no effect on global average temperatures."
         6. "Calcium channels are affected by AMP."
         7. "People who drink soda are much healthier than those who don't."

        Examples of non-claims (these are not valid scientific claims):
         1. "The sky is beautiful." (This is an opinion, not a testable claim)
         2. "What is the effect of exercise on heart health?" (This is a question, not a claim)
         3. "Scientists should study climate change more." (This is a recommendation, not a claim)
         4. "Drink more water!" (This is a command, not a claim),
         5. "Investigating the cognitive effects of BRCA2 mutations on intelligence quotient (IQ) levels." (This doesn't make a claim about anything)

        Reject claims that include ambiguous abbreviations or shorthand, unless it's clear to you what they mean. Remember, a valid scientific claim should be a specific, testable assertion about a phenomenon or relationship between variables. It doesn't have to be true, but it should be a testable assertion.

        For the 'suggested' field, focus on using clear, concise language with relevant scientific terms that would be 
        likely to appear in academic papers. Avoid colloquialisms and ensure the suggested version maintains the 
        original meaning (even if you think it's not true) while being more search-friendly.
        """

        for claim in claims:
            prompt = f"Evaluate if the following is a valid scientific claim and suggest an optimized version for search:\n\n{claim}"
            
            messages = [
                {"role": "system", "content": system_prompt or default_system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            task = self._make_request(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
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
                        'is_valid': False,
                        'explanation': f"Error: {str(response)}",
                        'suggested': original_claim
                    })
                    continue
                    
                self._update_token_usage(response.usage)
                result = json.loads(response.choices[0].message.content)
                result['original'] = original_claim
                results.append(result)
            
            return results
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            raise

    async def generate_json_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o") -> Any:
        # Check that the prompt is not too long - if it is, return a failure message
        if len(prompt) + len(system_prompt or "") > 320000:
            return json.loads('{"error": "Prompt is too long"}')

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant. Please provide your response in valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        print(f"Sending message to OpenAI.")
        
        response = await self._make_request(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        self._update_token_usage(response.usage)
        logger.info(f"API call completed for model {model}")
        return json.loads(response.choices[0].message.content)
    
    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o") -> str:
        # Check that the prompt is not too long - if it is, return a failure message
        if len(prompt) + len(system_prompt or "") > 320000:
            return "Error: Prompt is too long"

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        print(f"Sending message to OpenAI.")
        response = await self._make_request(model=model, messages=messages, temperature=0.0)
        self._update_token_usage(response.usage)
        logger.info(f"API call completed for model {model}")
        return response.choices[0].message.content
