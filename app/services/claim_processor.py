from typing import List, Union
import pandas as pd
from app.models.claim import Claim
from app.models.batch_job import BatchJob
from app.models.paper import Paper
from app.services.literature_searcher import LiteratureSearcher
from app.services.paper_analyzer import PaperAnalyzer
from app.services.evidence_scorer import EvidenceScorer
from app.services.openai_service import OpenAIService
import os
import json
import time as time_module
from time import time
from textwrap import dedent
from typing import Dict
from datetime import datetime
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimProcessor:
    def __init__(self):
        self.literature_searcher = LiteratureSearcher()
        self.paper_analyzer = PaperAnalyzer()
        self.evidence_scorer = EvidenceScorer()
        self.openai_service = OpenAIService()
        self.timing_stats = {}

    async def process_claim_with_papers(self, papers: List[Paper], batch_id: str, claim_id: str, claim_text: str) -> None:
        """Process a claim when we already have the papers."""
        start_time = time()
        timing_stats = {}

        try:
            if not papers:
                report = {
                    "relevantPapers": [],
                    "explanation": "No relevant papers were found for this claim.",
                    "claimRating": 0,
                    "timing_stats": timing_stats,
                    "searchQueries": self.literature_searcher.saved_search_queries,
                    "claim_text": claim_text
                }
                self.update_claim_status(batch_id, claim_id, "processed", report, claim_text)
                return

            # Process papers
            processed_papers = []
            non_relevant_papers = []
            inaccessible_papers = []
            total_papers = len(papers)
            
            paper_processing_start = time()
            logger.info("Processing papers")
            
            # Process papers concurrently with a semaphore to limit concurrent API calls
            sem = asyncio.Semaphore(2)  # Limit concurrent paper processing
            
            async def process_single_paper(paper, i):
                async with sem:
                    try:
                        # Fetch paper content
                        content, content_type = await self.literature_searcher.fetch_paper_content(paper)
                        
                        if not content:
                            inaccessible_papers.append({
                                'paper': paper,
                                'reason': 'Content not available'
                            })
                            return

                        # Analyze relevance
                        logger.info("Analyzing relevance")
                        relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages = (
                            await self.paper_analyzer.analyze_relevance_and_extract(content, claim_text)
                        )
                        
                        if relevance >= 0.1:
                            # Calculate paper weight score
                            weight_score = await self.evidence_scorer.calculate_paper_weight(paper)
                            
                            processed_papers.append({
                                'paper': paper,
                                'relevance': relevance,
                                'excerpts': excerpts,
                                'score': weight_score,
                                'explanations': explanations,
                                'content_type': content_type,
                                'excerpt_pages': excerpt_pages
                            })
                            logger.info("Added to processed papers")
                        else:
                            non_relevant_papers.append({
                                'paper': paper,
                                'explanation': non_relevant_explanation,
                                'content_type': content_type
                            })
                            logger.info("Added to non relevant papers")
                    except Exception as e:
                        logger.error(f"Error processing paper {paper.paper_id}: {str(e)}")

            # Process all papers concurrently
            paper_tasks = [process_single_paper(paper, i) for i, paper in enumerate(papers)]
            await asyncio.gather(*paper_tasks)

            timing_stats['paper_processing'] = time() - paper_processing_start

            # Generate final report
            logger.info("Generating report")
            report_start = time()
            
            if processed_papers:
                logger.info("Processed papers is not empty")
                report = await self.generate_final_report(
                    claim_text,
                    processed_papers, 
                    non_relevant_papers, 
                    inaccessible_papers
                )
                logger.info("Generated report")
                
                # Ensure claim text is in report
                report["claim_text"] = claim_text
                
                timing_stats['report_generation'] = time() - report_start
                timing_stats['total_time'] = time() - start_time
                report['timing_stats'] = timing_stats
                
                self.update_claim_status(batch_id, claim_id, "processed", report, claim_text)
            else:
                report = {
                    "relevantPapers": [],
                    "nonRelevantPapers": self._format_non_relevant_papers(non_relevant_papers),
                    "inaccessiblePapers": self._format_inaccessible_papers(inaccessible_papers),
                    "explanation": "No relevant papers were found that support or refute this claim.",
                    "claimRating": 0,
                    "timing_stats": timing_stats,
                    "searchQueries": self.literature_searcher.saved_search_queries,
                    "claim_text": claim_text
                }
                self.update_claim_status(batch_id, claim_id, "processed", report, claim_text)

        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            report = {
                "error": str(e),
                "timing_stats": timing_stats,
                "claim_text": claim_text
            }
            self.update_claim_status(batch_id, claim_id, "error", report, claim_text)

    def _format_non_relevant_papers(self, papers: List[Dict]) -> List[Dict]:
        """Format non-relevant papers for the report."""
        try:
            return [{
                "title": paper.get('paper', {}).title,
                "authors": [
                    {
                        "name": author.get('name', 'Unknown'),
                        "hIndex": author.get('hIndex', 0)
                    }
                    for author in (paper.get('paper', {}).authors or [])
                ],
                "link": paper.get('paper', {}).url,
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
                "title": paper['paper'].title,
                "authors": [
                {
                    "name": author['name'],
                    "hIndex": author.get('hIndex', 0)
                }
                for author in paper['paper'].authors
            ],
                "link": paper['paper'].url,
                "reason": paper['reason']
            } for paper in papers]
        except Exception as e:
            print("Error formatting inaccessible papers: ", e)
            return []

    async def generate_final_report(self, claim_text: str, processed_papers: List[dict], 
                                  non_relevant_papers: List[dict], 
                                  inaccessible_papers: List[dict]) -> dict:
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
                        
                    authors_str = ', '.join(
                        f"{author.get('name', 'Unknown')} (H-index: {author.get('hIndex', 0)})"
                        for author in (p['paper'].authors or [])
                    )
                    
                    summary = (
                        f"Paper: {p['paper'].title}\n"
                        f"Authors: {authors_str}\n"
                        f"Relevance: {p.get('relevance', 'Unknown')}\n"
                        f"Reliability Weight: {p.get('score', 'Unknown')}\n"
                        f"Excerpts: {p.get('excerpts', [])}"
                    )
                    paper_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error processing paper summary: {str(e)}")
                    continue

            paper_summaries_text = "\n".join(paper_summaries)

            # Prepare input for the LLM
            prompt = dedent(f"""
            Evaluate the following claim based on the provided evidence and counter-evidence from scientific papers:

            Claim: {claim_text}

            Evidence:
            {paper_summaries_text}
            """).strip()

            system_prompt = dedent("""
            You are an expert scientific reviewer specializing in evaluating the plausibility of scientific claims based on evidence from academic papers and your expert knowledge. Your task is to synthesize a detailed evaluation of the claim and assign a final plausibility rating based on both your scientific knowledge and the evidence provided in the paper excerpts you receive.

            The final rating you assign should be one of the following:
            - Contradicted: Strong evidence refutes the claim.
            - Implausible: Evidence suggests the claim is highly unlikely but not definitively refuted.
            - No Evidence: No significant evidence is available to support or refute the claim.
            - Little Evidence: Limited or weak evidence suggests the claim may be plausible but is not conclusive.
            - Plausible: The claim is supported by reasonable evidence, though it may not be definitive.
            - Highly Supported: The claim is strongly supported by compelling and consistent evidence.
            
            When formulating your evaluation, consider the following aspects:
            - Supporting Evidence: Summarize the most robust evidence that supports the claim. Be specific, referencing the findings of relevant papers and their implications.
            - Caveats or Contradictions: Identify any limitations, contradictory findings, or alternative interpretations that might challenge the claim.
            - Analysis: Based on your expertise, analyze the systems and structures relevant to the claim for any deeper relationships, mechanisms, or second-order implications that might be relevant.
            - Assessment: Assess the balance of evidence, explaining which side is more compelling and why. Contextualize caveats but avoid undue hedging; consider the overall weight of the evidence like an expert would.
            - Rating Assignment: Choose a single category from the list above that best reflects the overall strength of evidence for the claim. Assign this rating based on the preponderance of evidence, contextualizing caveats without allowing minor exceptions to overshadow the dominant trend.
            
            Write the explanation as an essay with distinct paragraphs for:
            - Supporting evidence.
            - Caveats or contradictory evidence.
            - Analysis of potential underlying mechanisms, deeper relationships, or second-order implications.
            - An explanation of which rating is most appropriate based on the relative strength of the evidence.
            
            You will receive the text of the claim and excerpts from academic papers that could support or refute the claim. Craft your evaluation, then provide a JSON response in the following format:
            {
                "explanationEssay": "<detailed essay explanation>",
                "claimRating": "<rating, one of the following: Contradicted, Implausible, No Evidence, Little Evidence, Plausible, Highly Supported>"
            }
            """).strip()

            logger.info("Generating final report with LLM")
            response = await self.openai_service.generate_json_async(prompt, system_prompt)
            logger.info("Generated final report")

            # Convert the claimRating to a number
            claimRating = 0
            if response.get('claimRating') == 'Contradicted':
                claimRating = 0
            elif response.get('claimRating') == 'Implausible':
                claimRating = 1
            elif response.get('claimRating') == 'No Evidence':
                claimRating = 2
            elif response.get('claimRating') == 'Little Evidence':
                claimRating = 3
            elif response.get('claimRating') == 'Plausible':
                claimRating = 4
            elif response.get('claimRating') == 'Highly Supported':
                claimRating = 5

            # Format the final report
            return {
                "relevantPapers": [
                    {
                        "title": p.get('paper', {}).title,
                        "authors": [
                            {
                                "name": author.get('name', 'Unknown'),
                                "hIndex": author.get('hIndex', 0)
                            }
                            for author in (p.get('paper', {}).authors or [])
                        ],
                        "link": p.get('paper', {}).url,
                        "relevance": p.get('relevance', 0),
                        "weight_score": p.get('score', 0),
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
                    for p in (processed_papers or [])
                    if p.get('paper')
                ],
                "nonRelevantPapers": self._format_non_relevant_papers(non_relevant_papers or []),
                "inaccessiblePapers": self._format_inaccessible_papers(inaccessible_papers or []),
                "explanation": response.get('explanationEssay', 'No explanation available'),
                "claimRating": claimRating,
                "searchQueries": getattr(self.literature_searcher, 'saved_search_queries', []),
                "usage_stats": self.openai_service.get_usage_stats()
            }

        except Exception as e:
            logger.error(f"Error in generate_final_report: {str(e)}")
            # Return a safe fallback response
            return {
                "relevantPapers": [],
                "nonRelevantPapers": [],
                "inaccessiblePapers": [],
                "explanation": f"Error generating final report: {str(e)}",
                "claimRating": 0,
                "searchQueries": [],
                "usage_stats": self.openai_service.get_usage_stats()
            }

    def _format_citation(self, paper, page_number):
        """Format citation in RIS format."""
        authors = ' and '.join([author['name'] for author in paper.authors])
        return f"""
        TY  - JOUR
        TI  - {paper.title}
        AU  - {authors}
        PY  - {paper.year}
        JO  - {paper.journal}
        UR  - {paper.url}
        SP  - {page_number}
        ER  -
        """.strip()

    def update_claim_status(self, batch_id: str, claim_id: str, status: str, report: dict = None, claim_text: str = None):
        """Update claim status and report in saved_jobs directory."""
        try:
            claim_dir = os.path.join('saved_jobs', batch_id)
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
