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

    async def process_claim_with_papers(self, papers: List[Paper], batch_id: str, claim_id: str) -> None:
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
                    "searchQueries": self.literature_searcher.saved_search_queries
                }
                self.update_claim_status(batch_id, claim_id, "processed", report)
                return

            # Process papers
            processed_papers = []
            non_relevant_papers = []
            inaccessible_papers = []
            total_papers = len(papers)
            
            paper_processing_start = time()
            logger.info("Processing papers")
            
            # Process papers concurrently with a semaphore to limit concurrent API calls
            sem = asyncio.Semaphore(3)  # Limit concurrent paper processing
            
            async def process_single_paper(paper, i):
                async with sem:
                    try:
                        # Fetch paper content
                        content, content_type = await self.literature_searcher.fetch_paper_content(paper, None)
                        
                        if not content:
                            inaccessible_papers.append({
                                'paper': paper,
                                'reason': 'Content not available'
                            })
                            return

                        # Analyze relevance (we'll need to modify paper_analyzer to be async)
                        logger.info("Analyzing relevance")
                        relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages = (
                            await self.paper_analyzer.analyze_relevance_and_extract(content, paper.title)
                        )
                        
                        if relevance >= 0.1:
                            # Calculate paper weight score
                            weight_score = self.evidence_scorer.calculate_paper_weight(paper)
                            
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
                    papers[0].text,  # Use the first paper's text as the claim text
                    processed_papers, 
                    non_relevant_papers, 
                    inaccessible_papers
                )
                logger.info("Generated report")
                
                timing_stats['report_generation'] = time() - report_start
                timing_stats['total_time'] = time() - start_time
                report['timing_stats'] = timing_stats
                
                self.update_claim_status(batch_id, claim_id, "processed", report)
            else:
                report = {
                    "relevantPapers": [],
                    "nonRelevantPapers": self._format_non_relevant_papers(non_relevant_papers),
                    "inaccessiblePapers": self._format_inaccessible_papers(inaccessible_papers),
                    "explanation": "No relevant papers were found that support or refute this claim.",
                    "claimRating": 0,
                    "timing_stats": timing_stats,
                    "searchQueries": self.literature_searcher.saved_search_queries
                }
                self.update_claim_status(batch_id, claim_id, "processed", report)

        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            report = {
                "error": str(e),
                "timing_stats": timing_stats
            }
            self.update_claim_status(batch_id, claim_id, "error", report)

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
            You are an AI assistant tasked with evaluating scientific claims based on evidence from papers.
            Provide an explanation in an essay format with newlines between paragraphs, including specific references to the scientific papers. 
            The essay should have:
            1. A paragraph highlighting supporting evidence
            2. A paragraph highlighting caveats or contradictions
            3. An analysis of which evidence outweighs the other and how strongly the claim is supported
            
            Assign a claim rating between -10 (completely refuted) and 10 (strongly supported):
            -10 to -7: Contradicted by strong evidence
            -6 to -4: Somewhat refuted
            -3 to -1: Slightly refuted
            0: No evidence either way
            1 to 3: Slightly supported
            4 to 6: Reasonably supported
            7 to 10: Strongly supported

            The JSON response should have:
            {
                "explanation": <str: detailed essay explaining the rating>,
                "claimRating": <int: rating from -10 to 10>
            }
            """).strip()

            logger.info("Generating final report with LLM")
            response = await self.openai_service.generate_json_async(prompt, system_prompt)
            logger.info("Generated final report")

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
                "explanation": response.get('explanation', 'No explanation available'),
                "claimRating": response.get('claimRating', 0),
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
            
            data = {
                "status": status,
                "text": claim_text,
                "additional_info": ""
            }
            if report:
                data["report"] = report
                
            with open(claim_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating claim status: {str(e)}")
