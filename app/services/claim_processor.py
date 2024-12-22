from typing import List, Union
import pandas as pd
from app.models.claim import Claim
from app.models.batch_job import BatchJob
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimProcessor:
    def __init__(self):
        self.literature_searcher = LiteratureSearcher()
        self.paper_analyzer = PaperAnalyzer()
        self.evidence_scorer = EvidenceScorer()
        self.openai_service = OpenAIService()
        self.timing_stats = {}

    def process_claim(self, claim: Claim, batch_id: str, claim_id: str) -> None:
        """Process a single claim."""
        start_time = time()
        timing_stats = {}

        try:
            self.update_claim_status(batch_id, claim_id, "searching_papers")
            
            # Search for relevant papers using S2
            search_start = time()
            papers = self.literature_searcher.search_papers(claim)
            print("Got papers")
            print("Papers: ", papers)
            timing_stats['search_papers'] = time() - search_start

            if not papers:
                claim.status = "processed"
                claim.report = {
                    "supportingPapers": [],
                    "explanation": "No relevant papers were found for this claim.",
                    "claimRating": 0,
                    "timing_stats": timing_stats,
                    "searchQueries": self.literature_searcher.saved_search_queries
                }
                self.update_claim_status(batch_id, claim_id, "processed", claim.report)
                return

            self.update_claim_status(batch_id, claim_id, "analyzing_papers")
            
            # Process papers
            processed_papers = []
            non_relevant_papers = []
            inaccessible_papers = []
            total_papers = len(papers)
            
            paper_processing_start = time()
            print("Processing papers")
            
            for i, paper in enumerate(papers):
                self.update_claim_status(
                    batch_id, 
                    claim_id, 
                    f"analyzing_paper_{i+1}_of_{total_papers}"
                )
                
                try:
                    # We already have the content in the paper object
                    content = paper.text
                    content_type = paper.content_source
                    
                    if not content:
                        inaccessible_papers.append({
                            'paper': paper,
                            'reason': 'Content not available'
                        })
                        continue

                    # Analyze relevance
                    print("Analyzing relevance")
                    relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages = (
                        self.paper_analyzer.analyze_relevance_and_extract(content, claim)
                    )
                    print("Analyzed relevance")
                    if relevance >= 0.1:
                        # Calculate paper weight score using S2 metrics
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
                        print("Added to processed papers")
                    else:
                        non_relevant_papers.append({
                            'paper': paper,
                            'explanation': non_relevant_explanation,
                            'content_type': content_type
                        })
                        print("Added to non relevant papers")
                except Exception as e:
                    logger.error(f"Error processing paper {paper.paper_id}: {str(e)}")
                    continue

            timing_stats['paper_processing'] = time() - paper_processing_start

            # Generate final report
            self.update_claim_status(batch_id, claim_id, "generating_report")
            report_start = time()
            print("Generating report")
            
            if processed_papers:
                print("Processed papers is not empty")
                claim.report = self.generate_final_report(
                    claim, processed_papers, non_relevant_papers, inaccessible_papers
                )
                print("Generated report")
                timing_stats['report_generation'] = time() - report_start
                timing_stats['total_time'] = time() - start_time
                claim.report['timing_stats'] = timing_stats
                claim.status = "processed"
                self.update_claim_status(batch_id, claim_id, "processed", claim.report)
            else:
                claim.status = "processed"
                claim.report = {
                    "supportingPapers": [],
                    "nonRelevantPapers": self._format_non_relevant_papers(non_relevant_papers),
                    "inaccessiblePapers": self._format_inaccessible_papers(inaccessible_papers),
                    "explanation": "No relevant papers were found that support or refute this claim.",
                    "claimRating": 0,
                    "timing_stats": timing_stats,
                    "searchQueries": self.literature_searcher.saved_search_queries
                }
                self.update_claim_status(batch_id, claim_id, "processed", claim.report)

        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            claim.status = "error"
            claim.report = {
                "error": str(e),
                "timing_stats": timing_stats
            }
            self.update_claim_status(batch_id, claim_id, "error", claim.report)

    def _format_non_relevant_papers(self, papers: List[Dict]) -> List[Dict]:
        """Format non-relevant papers for the report."""
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
            "explanation": paper['explanation'],
            "content_type": paper['content_type']
        } for paper in papers]

    def _format_inaccessible_papers(self, papers: List[Dict]) -> List[Dict]:
        """Format inaccessible papers for the report."""
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

    def generate_final_report(self, claim: Claim, processed_papers: List[dict], 
                            non_relevant_papers: List[dict], 
                            inaccessible_papers: List[dict]) -> dict:
        """Generate the final report for a claim."""
        # Prepare input for the LLM
        paper_summaries = "\n".join([
            f"Paper: {p['paper'].title}\n"
            f"Authors: {', '.join(author['name'] + ' (H-index: ' + str(author.get('hIndex', 0)) + ')' for author in p['paper'].authors)}\n"
            f"Relevance: {p['relevance']}\nReliability Weight: {p['score']}\nExcerpts: {p['excerpts']}"
            for p in processed_papers
        ])
        
        prompt = dedent(f"""
        Evaluate the following claim based on the provided evidence from scientific papers:

        Claim: {claim.text}

        Evidence:
        {paper_summaries}
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

        print("Prompt: ", prompt)

        response = self.openai_service.generate_json(prompt, system_prompt)

        # Format the final report
        try:
            return {
                "supportingPapers": [
                    {
                    "title": p['paper'].title,
                    "authors": [
                        {
                            "name": author['name'],
                            "hIndex": author.get('hIndex', 0)
                        }
                        for author in p['paper'].authors
                    ],
                    "link": p['paper'].url,
                    "relevance": p['relevance'],
                    "weight_score": p['score'],
                    "content_type": p['content_type'],
                    "excerpts": p['excerpts'],
                    "explanations": p['explanations'],
                    "citations": [
                        {
                            "text": excerpt,
                            "page": page,
                            "citation": self._format_citation(p['paper'], page)
                        }
                        for excerpt, page in zip(p['excerpts'], p.get('excerpt_pages', []))
                    ]
                }
                for p in processed_papers
            ],
            "nonRelevantPapers": self._format_non_relevant_papers(non_relevant_papers),
            "inaccessiblePapers": self._format_inaccessible_papers(inaccessible_papers),
            "explanation": response['explanation'],
            "claimRating": response['claimRating'],
            "searchQueries": self.literature_searcher.saved_search_queries,
                "usage_stats": self.openai_service.get_usage_stats()
            }
        except Exception as e:
            print("Error generating final report: ", e)
            return {
                "supportingPapers": [],
                "nonRelevantPapers": [],
                "inaccessiblePapers": [],
                "explanation": "Error generating final report",
                "claimRating": 0,
                "searchQueries": self.literature_searcher.saved_search_queries,
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

    def update_claim_status(self, batch_id: str, claim_id: str, status: str, report: dict = None):
        """Update claim status and report in saved_jobs directory."""
        try:
            claim_dir = os.path.join('saved_jobs', batch_id)
            os.makedirs(claim_dir, exist_ok=True)
            claim_file = os.path.join(claim_dir, f"{claim_id}.txt")
            
            data = {
                "status": status,
                "additional_info": ""
            }
            if report:
                data["report"] = report
                
            with open(claim_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating claim status: {str(e)}")
