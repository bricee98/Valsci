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
        Your task is to analyze the given paper content, assess the relevance to the claim (including whether it supports or refutes the claim), extract relevant verbatim sections, and then score the overall relevance to the provided claim.
        Provide your analysis in a structured JSON format.
        """).strip()

        user_prompt = dedent(f"""
        First, analyze the paper content and determine whether it supports or refutes the given claim. Use a deep understanding of the processes and interactions involved.
        Then, extract verbatim sections from the paper that support or refute the given claim.
        Finally, score the relevance of the paper content to the claim based on these excerpts.
        
        Claim: {claim.text}
        
        Paper content:
        {paper_content}
        
        Respond with a JSON object containing:
        1. 'explanations': A list of explanations for why the excerpts are relevant to the claim
        2. 'excerpts': A list of relevant verbatim excerpts from the paper
        3. 'relevance': A float between 0 (not relevant) and 1 (highly relevant), based on the extracted excerpts
        """).strip()
        
        result = self.openai_service.generate_json(user_prompt, system_prompt)

        print("Result: ", result)
        print("Excerpts: ", result['excerpts'])
        
        return result['relevance'], result['excerpts'], result['explanation']
