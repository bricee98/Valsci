from typing import List, Tuple, Optional
from app.models.claim import Claim
from app.services.openai_service import OpenAIService
from textwrap import dedent
import logging
import re

logger = logging.getLogger(__name__)

class PaperAnalyzer:
    def __init__(self):
        self.openai_service = OpenAIService()

    def analyze_relevance_and_extract(
        self, 
        paper_content: str, 
        claim: Claim
    ) -> Tuple[float, List[str], List[str], Optional[str], List[int]]:
        """
        Analyze paper content for relevance to the claim and extract supporting evidence.
        
        Returns:
        - relevance score (0-1)
        - list of relevant excerpts
        - list of explanations for each excerpt
        - explanation if paper is not relevant
        - list of page numbers for excerpts
        """
        
        # Clean and format the content
        cleaned_content = self._clean_content(paper_content)
        
        # Prepare the analysis prompt
        system_prompt = dedent("""
            You are an expert at analyzing scientific papers and determining their relevance to specific claims.
            Analyze the given paper content and determine if it contains evidence that supports or refutes the claim.
            
            Guidelines:
            1. Focus on direct evidence, not tangential relationships
            2. Look for specific methods, results, and conclusions
            3. Consider both supporting and contradicting evidence
            4. Identify exact quotes that are most relevant
            5. Note the context and limitations of the evidence
            
            Return a JSON object with:
            {
                "relevance": float (0-1),
                "excerpts": list of relevant verbatim quotes,
                "explanations": list of explanations for each excerpt,
                "non_relevant_explanation": string (only if relevance < 0.1),
                "excerpt_pages": list of page numbers (or null if not available)
            }
        """).strip()

        user_prompt = dedent(f"""
            Analyze this paper content for evidence related to the following claim:
            
            Claim: {claim.text}
            
            Paper content:
            {cleaned_content}
            
            Determine if this paper provides relevant evidence for or against the claim.
            Extract verbatim quotes that directly support or refute the claim.
            Explain how each excerpt relates to the claim.
            If the paper is not relevant (relevance < 0.1), explain why.
        """).strip()

        try:
            result = self.openai_service.generate_json(user_prompt, system_prompt)
            
            # Log the analysis results
            logger.info(f"Paper analysis results:")
            logger.info(f"- Relevance: {result.get('relevance', 0)}")
            logger.info(f"- Number of excerpts: {len(result.get('excerpts', []))}")
            
            if result.get('relevance', 0) < 0.1:
                logger.info(f"- Not relevant: {result.get('non_relevant_explanation')}")
            
            return (
                result.get('relevance', 0),
                result.get('excerpts', []),
                result.get('explanations', []),
                result.get('non_relevant_explanation'),
                result.get('excerpt_pages', [])
            )

        except Exception as e:
            logger.error(f"Error analyzing paper content: {str(e)}")
            return 0, [], [], "Error analyzing paper content", []

    def _clean_content(self, content: str) -> str:
        """Clean and format paper content for analysis."""
        if not content:
            return ""
        
        # Remove multiple newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Clean up common OCR artifacts
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', content)
        
        # Remove references section if present
        if 'References' in content:
            content = content.split('References')[0]
        elif 'REFERENCES' in content:
            content = content.split('REFERENCES')[0]
            
        # Truncate if too long (for API limits)
        max_length = 12000  # Adjust based on your needs
        if len(content) > max_length:
            content = content[:max_length] + "...[truncated]"
        
        return content.strip()

    def extract_page_numbers(self, content: str, excerpt: str) -> Optional[int]:
        """
        Attempt to extract page numbers for excerpts.
        Returns None if page number cannot be determined.
        """
        try:
            # Look for page markers in the content
            page_markers = re.finditer(r'(?i)page\s*(\d+)|pg\.\s*(\d+)|\[(\d+)\]', content)
            excerpt_pos = content.find(excerpt)
            
            if excerpt_pos == -1:
                return None
                
            # Find the closest page marker before the excerpt
            closest_page = None
            closest_distance = float('inf')
            
            for match in page_markers:
                page_num = next(num for num in match.groups() if num is not None)
                distance = excerpt_pos - match.start()
                
                if 0 <= distance < closest_distance:
                    closest_distance = distance
                    closest_page = int(page_num)
            
            return closest_page
            
        except Exception as e:
            logger.error(f"Error extracting page number: {str(e)}")
            return None
