from typing import List, Tuple, Optional
from app.models.claim import Claim
from app.services.openai_service import OpenAIService
from textwrap import dedent
import logging
import re

logger = logging.getLogger(__name__)

class PaperAnalyzer:

    async def analyze_relevance_and_extract(
        self, 
        paper_content: str, 
        claim_text: str,
        ai_service
    ) -> Tuple[float, List[str], List[str], Optional[str], List[int]]:
        """
        Analyze paper content for relevance to the claim and extract supporting, contradicting, or generally relevant evidence.
        
        Returns:
        - relevance score (0-1)
        - list of relevant excerpts
        - list of explanations for each excerpt
        - explanation if paper is not relevant
        - list of page numbers for excerpts
        - usage stats
        """
        
        # Clean and format the content
        cleaned_content = self._clean_content(paper_content)
        
        # Prepare the analysis prompt
        system_prompt = dedent("""
            You are an expert at analyzing scientific papers and evaluating their relevance to specific claims through both direct evidence and mechanistic pathways.

            Guidelines for Analysis:
            1. Evaluate direct evidence that supports or refutes the claim
            2. Identify mechanistic evidence that strengthens or weakens the claim's plausibility
            3. Examine methodology, results, and conclusions with careful attention to detail
            4. Extract verbatim quotes with complete scientific context in which they are found
            5. Consider study limitations and their impact on evidence quality
            6. Assess both statistical and practical significance of findings
            7. Note experimental conditions that may affect generalizability

            Guidelines for Quote Extraction:
            1. Include complete sentences or paragraphs that capture full context
            2. Maintain exact spelling, punctuation, and formatting

            Return a JSON object with:
            {
                "relevance": float (0-1),
                "excerpts": list of relevant verbatim sentences or paragraphs (a list of strings),
                "explanations": list of explanations (a list of strings), one for each excerpt, addressing how the excerpt relates to the claim (direct or mechanistic) and the strength and limitations of the evidence
                "non_relevant_explanation": string (only if relevance < 0.1),
                "excerpt_pages": list of page numbers (or null if not available)
            }
        """).strip()

        user_prompt = dedent(f"""
            Analyze this paper content for both direct and mechanistic evidence related to the following claim:

            Claim: {claim_text}

            Paper content:
            {cleaned_content}

            Tasks:
            1. Determine if this paper provides relevant evidence for or against the claim
            2. Extract complete, verbatim sentences or paragraphs that:
            - Support or refute the claim
            - Describe relevant mechanisms
            - Provide essential context for understanding the evidence
            3. For each sentence or paragraph:
            - Explain how it relates to the claim
            - Note whether it's direct evidence or mechanistic
            - Include any limitations or caveats
            4. If relevance < 0.1, provide a detailed explanation why

            Remember:
            - Include complete sentences and surrounding context
            - Maintain exact wording, including statistical details
        """).strip()

        try:
            # Use the async version of generate_json
            result = await ai_service.generate_json_async(user_prompt, system_prompt)
            response = result['content']
            usage = result['usage']
            
            # Log the analysis results
            logger.info(f"Paper analysis results:")
            logger.info(f"- Relevance: {response.get('relevance', 0)}")
            logger.info(f"- Number of excerpts: {len(response.get('excerpts', []))}")
            
            if response.get('relevance', 0) < 0.1:
                logger.info(f"- Not relevant: {response.get('non_relevant_explanation')}")
            
            return (
                response.get('relevance', 0),
                response.get('excerpts', []),
                response.get('explanations', []),
                response.get('non_relevant_explanation'),
                response.get('excerpt_pages', []),
                usage
            )

        except Exception as e:
            logger.error(f"Error analyzing paper content: {str(e)}")
            # Return empty usage stats in error case
            return 0, [], [], "Error analyzing paper content", [], {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}

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
