from typing import List, Tuple
from app.models.claim import Claim
from app.services.openai_service import OpenAIService
from textwrap import dedent

class PaperAnalyzer:
    def __init__(self):
        self.openai_service = OpenAIService()

    def analyze_relevance_and_extract(self, paper_content: str, claim: Claim) -> Tuple[float, List[str]]:
        system_prompt = dedent("""
        You are an expert in analyzing scientific papers and determining their relevance to specific claims.
        Your task is to analyze the given paper content, extract relevant verbatim sections, and then assess its overall relevance to the provided claim.
        Provide your analysis in a structured JSON format.
        """).strip()

        user_prompt = dedent(f"""
        First, extract verbatim sections from the paper that support or refute the given claim.
        Then, analyze the relevance of the paper content to the claim based on these excerpts.
        
        Claim: {claim.text}
        
        Paper content:
        {paper_content}
        
        Respond with a JSON object containing:
        1. 'excerpts': A list of relevant verbatim excerpts from the paper
        2. 'relevance': A float between 0 (not relevant) and 1 (highly relevant), based on the extracted excerpts
        """).strip()
        
        result = self.openai_service.generate_json(user_prompt, system_prompt)
        
        return result['relevance'], result['excerpts']
