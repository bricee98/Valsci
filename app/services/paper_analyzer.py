from typing import List, Tuple, Optional
from app.models.claim import Claim
from app.services.llm.gateway import LLMTask
from app.services.llm.types import empty_usage
from textwrap import dedent
import logging
import re

logger = logging.getLogger(__name__)

class PaperAnalyzer:

    async def analyze_relevance_and_extract(
        self, 
        paper_content: str, 
        claim_text: str,
        ai_service,
        batch_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        model_override: Optional[str] = None,
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

        cleaned_content, truncation = self._clean_content_with_budget(
            content=paper_content,
            claim_text=claim_text,
            system_prompt=system_prompt,
            ai_service=ai_service,
            model_hint=model_override,
            reserved_output_tokens=1200,
        )

        if truncation.get("truncated"):
            await ai_service.add_issue(
                batch_id=batch_id,
                claim_id=claim_id,
                severity="WARN",
                stage=LLMTask.PAPER_ANALYSIS,
                message="Paper content truncation applied to fit context window.",
                details={
                    "paper_id": paper_id,
                    "original_tokens": truncation.get("original_tokens"),
                    "kept_tokens": truncation.get("kept_tokens"),
                },
            )

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
            result = await ai_service.chat_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                task=LLMTask.PAPER_ANALYSIS,
                batch_id=batch_id,
                claim_id=claim_id,
                paper_id=paper_id,
                model_override=model_override,
            )
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
            return 0, [], [], "Error analyzing paper content", [], empty_usage(is_estimated=True)

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
        
        return content.strip()

    def _clean_content_with_budget(
        self,
        *,
        content: str,
        claim_text: str,
        system_prompt: str,
        ai_service,
        model_hint: Optional[str],
        reserved_output_tokens: int,
    ) -> Tuple[str, dict]:
        cleaned = self._clean_content(content)
        model_name = model_hint or getattr(ai_service, "default_model", "gpt-4o")
        estimator = getattr(ai_service, "token_estimator", None)
        if estimator is None:
            return cleaned, {"truncated": False, "original_tokens": 0, "kept_tokens": 0}

        base_prompt = dedent(f"""
            Analyze this paper content for both direct and mechanistic evidence related to the following claim:

            Claim: {claim_text}

            Paper content:
        """).strip()

        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_prompt},
        ]
        base_tokens = estimator.estimate_chat_tokens(base_messages, model_name)

        registry = getattr(ai_service, "model_registry", None)
        context_window = 128000
        if registry is not None:
            context_window = registry.context_window(model_name)

        safety_margin = int(getattr(ai_service, "context_safety_margin_tokens", 256))
        available_for_content = max(0, context_window - base_tokens - reserved_output_tokens - safety_margin)

        original_tokens = estimator.estimate_text_tokens(cleaned, model_name)
        if original_tokens <= available_for_content:
            return cleaned, {
                "truncated": False,
                "original_tokens": original_tokens,
                "kept_tokens": original_tokens,
            }

        if available_for_content <= 0:
            return "", {
                "truncated": True,
                "original_tokens": original_tokens,
                "kept_tokens": 0,
            }

        # Approximate truncation by token ratio to avoid expensive iterative slicing.
        ratio = available_for_content / max(1, original_tokens)
        keep_chars = max(200, int(len(cleaned) * ratio))
        truncated = cleaned[:keep_chars].rstrip() + "...[truncated]"
        kept_tokens = estimator.estimate_text_tokens(truncated, model_name)
        return truncated, {
            "truncated": True,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
        }

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
