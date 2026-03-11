from typing import List, Union
import pandas as pd
from app.models.claim import Claim
from app.models.batch_job import BatchJob
from app.models.paper import Paper
from semantic_scholar.utils.searcher import S2Searcher
from app.services.paper_analyzer import PaperAnalyzer
from app.services.evidence_scorer import EvidenceScorer
from app.services.llm.gateway import LLMTask
from app.services.llm.types import empty_usage
from app.services.llm.validators import validate_final_report_payload
from app.services.prompt_store import load_prompt, render_prompt
import os
import json
from time import time
from typing import Dict
from datetime import datetime
import logging
import asyncio
from app.config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimProcessor:
    def _format_non_relevant_papers(self, papers: List[Dict]) -> List[Dict]:
        """Format non-relevant papers for the report."""
        try:
            return [{
                "title": paper['paper'].get('title', 'Unknown Title'),
                "authors": [
                    {
                        "name": author.get('name', 'Unknown'),
                        "hIndex": author.get('hIndex', 0)
                    }
                    for author in paper['paper'].get('authors', [])
                ],
                "link": paper['paper'].get('url'),
                "explanation": paper.get('explanation', 'No explanation available'),
                "content_type": paper.get('content_type', 'unknown')
            } for paper in (papers or [])]
        except Exception as e:
            logger.error(f"Error formatting non-relevant papers: {str(e)}")
            return []

    def _format_inaccessible_papers(self, papers: List[Dict]) -> List[Dict]:
        """Format inaccessible papers for the report."""
        try:
            return [{
                "title": paper.get('title', 'Unknown Title'),
                "authors": [
                    {
                        "name": author.get('name', 'Unknown'),
                        "hIndex": author.get('hIndex', 0)
                    }
                    for author in paper.get('authors', [])
                ],
                "link": paper.get('url'),
                "reason": paper.get('access_reason', 'Paper content not accessible'),
                "reason_code": paper.get('access_reason_code', 'unknown'),
                "access_details": paper.get('access_details', {}),
            } for paper in (papers or [])]
        except Exception as e:
            logger.error(f"Error formatting inaccessible papers: {str(e)}")
            return []

    async def generate_final_report(self, claim_text: str, processed_papers: List[dict], 
                                  non_relevant_papers: List[dict], 
                                  inaccessible_papers: List[dict],
                                  queries: List[str],
                                  ai_service,
                                  bibliometric_config=None,
                                  batch_id: str = None,
                                  claim_id: str = None,
                                  model_override: str = None) -> dict:
        """Generate the final report for a claim."""
        try:
            # Debug logging
            logger.info(f"Processed papers: {len(processed_papers) if processed_papers else 'None'}")
            
            # Safely process paper summaries
            paper_summaries = []
            for p in (processed_papers or []):
                try:
                    if not p.get('paper'):
                        logger.error(f"Missing paper object in processed paper: {p}")
                        continue
                        
                    paper_data = p['paper']
                    authors_str = ', '.join(
                        f"{author.get('name', 'Unknown')} (H-index: {author.get('hIndex', 0)})"
                        for author in paper_data.get('authors', [])
                    )
                    
                    # Check if bibliometrics are enabled
                    use_bibliometrics = True
                    if bibliometric_config and 'use_bibliometrics' in bibliometric_config:
                        use_bibliometrics = bibliometric_config.get('use_bibliometrics')
                    
                    if use_bibliometrics:
                        summary = (
                            f"Paper: {paper_data.get('title', 'Unknown Title')}\n"
                            f"Authors: {authors_str}\n"
                            f"Relevance: {p.get('relevance', 'Unknown')}\n"
                            f"Bibliometric Impact: {p.get('score', 'Unknown')}\n"
                            f"Excerpts: {p.get('excerpts', [])}"
                        )
                    else:
                        summary = (
                            f"Paper: {paper_data.get('title', 'Unknown Title')}\n"
                            f"Authors: {authors_str}\n"
                            f"Relevance: {p.get('relevance', 'Unknown')}\n"
                            f"Excerpts: {p.get('excerpts', [])}"
                        )
                    paper_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error processing paper summary: {str(e)}")
                    continue

            paper_summaries_text = "\n".join(paper_summaries)

            # Prepare input for the LLM
            prompt_template = render_prompt("final_report_user", claim_text=claim_text)
            system_prompt = load_prompt("final_report_system")

            model_name = model_override or getattr(ai_service, "default_model", None)
            prompt = await self._build_final_prompt_with_budget(
                ai_service=ai_service,
                model_name=model_name,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                evidence_text=paper_summaries_text,
                batch_id=batch_id,
                claim_id=claim_id,
            )

            logger.info("Generating final report with LLM")
            result = await ai_service.chat_json(
                user_prompt=prompt,
                system_prompt=system_prompt,
                task=LLMTask.FINAL_REPORT,
                batch_id=batch_id,
                claim_id=claim_id,
                model_override=model_override,
            )
            response = validate_final_report_payload(result['content'])
            usage = result['usage']
            logger.info("Generated final report")

            # Convert the claimRating to a number
            claimRating = 0
            if response.get('claimRating') == 'Contradicted':
                claimRating = 1
            elif response.get('claimRating') == 'Likely False':
                claimRating = 2
            elif response.get('claimRating') == 'Mixed Evidence':
                claimRating = 3
            elif response.get('claimRating') == 'Likely True':
                claimRating = 4
            elif response.get('claimRating') == 'Highly Supported':
                claimRating = 5

            # Determine whether to include bibliometric impact
            use_bibliometrics = True
            if bibliometric_config and 'use_bibliometrics' in bibliometric_config:
                use_bibliometrics = bibliometric_config.get('use_bibliometrics')

            # Format the final report
            relevant_papers = []
            for p in (processed_papers or []):
                if not p.get('paper'):
                    continue
                    
                paper_info = {
                    "title": p['paper'].get('title', 'Unknown Title'),
                    "authors": [
                        {
                            "name": author.get('name', 'Unknown'),
                            "hIndex": author.get('hIndex', 0)
                        }
                        for author in p['paper'].get('authors', [])
                    ],
                    "link": p['paper'].get('url'),
                    "relevance": p.get('relevance', 0),
                    "content_type": p.get('content_type', 'unknown'),
                    "excerpts": p.get('excerpts', []),
                    "explanations": p.get('explanations', []),
                    "citations": [
                        {
                            "text": excerpt,
                            "page": page,
                            "citation": self._format_citation(p['paper'], page)
                        }
                        for excerpt, page in zip(
                            p.get('excerpts', []), 
                            p.get('excerpt_pages', []) or [None] * len(p.get('excerpts', []))
                        )
                    ]
                }
                
                # Add bibliometric impact (renamed from weight_score)
                if use_bibliometrics:
                    paper_info["bibliometric_impact"] = p.get('score', 0)
                
                relevant_papers.append(paper_info)

            return {
                "relevantPapers": relevant_papers,
                "nonRelevantPapers": self._format_non_relevant_papers(non_relevant_papers or []),
                "inaccessiblePapers": self._format_inaccessible_papers(inaccessible_papers or []),
                "explanation": response.get('explanationEssay', 'No explanation available'),
                "finalReasoning": response.get('finalReasoning', 'No additional reasoning available'),
                "claimRating": claimRating,
                "searchQueries": queries,
                "usage_stats": {},
                "bibliometric_config": bibliometric_config
            }, usage

        except Exception as e:
            logger.error(f"Error in generate_final_report: {str(e)}")
            if batch_id and claim_id and hasattr(ai_service, "add_issue"):
                await ai_service.add_issue(
                    batch_id=batch_id,
                    claim_id=claim_id,
                    severity="ERROR",
                    stage=LLMTask.FINAL_REPORT,
                    message="Final report output failed validation or generation.",
                    details={"error": str(e)},
                )
            # Return a safe fallback response with usage stats
            return {
                "relevantPapers": [],
                "nonRelevantPapers": [],
                "inaccessiblePapers": [],
                "explanation": f"Error generating final report: {str(e)}",
                "claimRating": -1,
                "searchQueries": [],
                "usage_stats": {},
                "bibliometric_config": bibliometric_config
            }, empty_usage(is_estimated=True)  # Add empty usage stats

    async def _build_final_prompt_with_budget(
        self,
        *,
        ai_service,
        model_name: str,
        system_prompt: str,
        prompt_template: str,
        evidence_text: str,
        batch_id: str,
        claim_id: str,
    ) -> str:
        estimator = getattr(ai_service, "token_estimator", None)
        registry = getattr(ai_service, "model_registry", None)
        if estimator is None or registry is None:
            return f"{prompt_template}\n{evidence_text}".strip()

        context_window = registry.context_window(model_name)
        reserved_output_tokens = 1800
        safety_margin = int(getattr(ai_service, "context_safety_margin_tokens", 256))
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template},
        ]
        base_tokens = estimator.estimate_chat_tokens(base_messages, model_name)
        available_evidence_tokens = max(0, context_window - base_tokens - reserved_output_tokens - safety_margin)
        evidence_tokens = estimator.estimate_text_tokens(evidence_text, model_name)

        if evidence_tokens <= available_evidence_tokens:
            return f"{prompt_template}\n{evidence_text}".strip()

        if available_evidence_tokens <= 0:
            await ai_service.add_issue(
                batch_id=batch_id,
                claim_id=claim_id,
                severity="ERROR",
                stage=LLMTask.FINAL_REPORT,
                message="Context overflow prevented for final report prompt.",
                details={
                    "estimated_tokens": evidence_tokens + base_tokens + reserved_output_tokens,
                    "context_limit": context_window,
                },
            )
            return prompt_template

        ratio = available_evidence_tokens / max(1, evidence_tokens)
        keep_chars = max(200, int(len(evidence_text) * ratio))
        truncated_evidence = evidence_text[:keep_chars].rstrip() + "...[truncated]"
        kept_tokens = estimator.estimate_text_tokens(truncated_evidence, model_name)
        await ai_service.add_issue(
            batch_id=batch_id,
            claim_id=claim_id,
            severity="WARN",
            stage=LLMTask.FINAL_REPORT,
            message="Final report evidence text truncation applied to fit context window.",
            details={
                "original_tokens": evidence_tokens,
                "kept_tokens": kept_tokens,
            },
        )
        return f"{prompt_template}\n{truncated_evidence}".strip()

    def _format_citation(self, paper, page_number):
        """Format citation in RIS format."""
        authors = ' and '.join([author.get('name', 'Unknown') for author in paper.get('authors', [])])
        return f"""
        TY  - JOUR
        TI  - {paper.get('title', 'Unknown Title')}
        AU  - {authors}
        PY  - {paper.get('year')}
        JO  - {paper.get('venue')}
        UR  - {paper.get('url')}
        SP  - {page_number}
        ER  -
        """.strip()

    def update_claim_status(self, batch_id: str, claim_id: str, status: str, report: dict = None, claim_text: str = None):
        """Update claim status and report in saved_jobs directory."""
        try:
            claim_dir = os.path.join(Config.SAVED_JOBS_DIR, batch_id)
            os.makedirs(claim_dir, exist_ok=True)
            claim_file = os.path.join(claim_dir, f"{claim_id}.txt")
            
            # Ensure claim_text is included in the data
            data = {
                "status": status,
                "text": claim_text,  # This was being set but not used when claim_text was None
                "additional_info": ""
            }
            if report:
                data["report"] = report
                # Also store claim text in report for consistency
                if claim_text:
                    data["report"]["claim_text"] = claim_text
                
            with open(claim_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating claim status: {str(e)}")
