import logging
import requests
import time
import random
from typing import List, Dict
from app.models.paper import Paper
from app.services.openai_service import OpenAIService
from requests.exceptions import RequestException
from textwrap import dedent
from app.config.settings import Config
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceScorer:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.email = Config.USER_EMAIL

    def calculate_paper_weight(self, paper: Paper) -> float:
        print("There are ", len(paper.authors), " authors")
        print("The authors are: ", paper.authors)
        print("Using first 4 authors")
        paper.authors = paper.authors[:4]

        max_h_index = max(self.get_author_h_index(author) for author in paper.authors)
        impact_factor_data = self.get_journal_impact_factor(paper.journal)
        
        impact_factor = impact_factor_data.get('impact_factor', 0) if isinstance(impact_factor_data, dict) else 0
        
        normalized_h_index = min(max_h_index / 50, 1)
        normalized_impact_factor = min(impact_factor / 20, 1)
        
        return (normalized_h_index + normalized_impact_factor) / 2

    def make_api_request(self, url: str, params: dict = None, headers: dict = None, method: str = 'get', max_retries: int = 7) -> requests.Response:
        if params is None:
            params = {}
        params['email'] = self.email

        # Construct the full URL with parameters
        full_url = url
        if params:
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query_string}"

        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request to {full_url} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                if method.lower() == 'get':
                    response = requests.get(url, params=params, headers=headers)
                elif method.lower() == 'post':
                    response = requests.post(url, json=params, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()

                if response.status_code == 429:
                    wait_time = min(15, max(5, (2 ** attempt) + (random.randint(0, 1000) / 1000)))
                    logger.warning(f"Rate limit reached for {full_url}. Status code: {response.status_code}. Waiting for {wait_time:.2f} seconds before retrying.")
                    time.sleep(wait_time)
                    continue
                
                time.sleep(3)  # Wait 3 seconds between requests
                return response
            except RequestException as e:
                if attempt == max_retries - 1:
                    status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'Unknown'
                    logger.error(f"Max retries reached for {full_url}. Status code: {status_code}. Error: {str(e)}")
                    raise e
                wait_time = min(60, (2 ** attempt) + (random.randint(0, 1000) / 1000))
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'Unknown'
                logger.warning(f"Request to {full_url} failed. Status code: {status_code}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    def get_author_h_index(self, author: Dict[str, str]) -> float:
        author_id = author.get('authorId')
        if not author_id:
            logger.warning(f"No author ID found for {author.get('name')}. Returning h-index of 0.")
            return 0

        url = f"https://api.openalex.org/authors/{author_id}"
        params = {"select": "summary_stats"}
        response = self.make_api_request(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            h_index = data.get('summary_stats', {}).get('h_index', 0)
            logger.info(f"Retrieved h-index for author {author.get('name')}: {h_index}")
            return h_index
        else:
            logger.warning(f"Failed to retrieve h-index for author {author.get('name')}. Returning 0.")
            return 0

    def get_journal_impact_factor(self, journal: str) -> dict:
        url = "https://api.openalex.org/sources"  # Changed from venues to sources
        params = {
            "filter": f"display_name.search:{journal}", 
            "select": "id,display_name,summary_stats"
        }
        response = self.make_api_request(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                source = data['results'][0]
                journal_info = f"Journal: {source['display_name']}, OpenAlex ID: {source['id']}"
                citation_count = source.get('summary_stats', {}).get('2yr_mean_citedness', 0)
                return self.estimate_impact_factor(journal_info, citation_count)
        return {'explanation': 'No data found.', 'impact_factor': 0}

    def estimate_impact_factor(self, journal_info: str, citation_count: float) -> dict:
        system_prompt = "You are an expert in academic publishing and journal metrics. Your task is to estimate the impact factor of a journal based on its information and citation count. Provide your response as a valid JSON object with 'explanation' and 'impact_factor' fields."
        
        prompt = dedent(f"""
        Given the following journal information and citation count, estimate the impact factor and provide a brief explanation. The impact factor should be a single number, typically ranging from 0 to 50 or more. Consider factors such as the journal's reputation, field of study, and historical impact.

        Journal information: {journal_info}
        2-year mean citedness: {citation_count}

        Respond with a valid JSON object in the following format:
        {{
            "explanation": "Your explanation here",
            "impact_factor": 0.0
        }}
        """).strip()

        response = self.openai_service.generate_json(prompt, system_prompt=system_prompt)
        try:
            explanation = response.get('explanation', 'No explanation provided.')
            impact_factor = float(response.get('impact_factor', 0))
            return {'explanation': explanation, 'impact_factor': impact_factor}
        except (ValueError, TypeError, KeyError):
            return {'explanation': 'Error in processing response.', 'impact_factor': 0}
