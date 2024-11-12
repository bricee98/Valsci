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

class ClaimProcessor:
    def __init__(self):
        self.literature_searcher = LiteratureSearcher()
        self.paper_analyzer = PaperAnalyzer()
        self.evidence_scorer = EvidenceScorer()
        self.openai_service = OpenAIService()
        self.timing_stats = {}  # Add timing stats dictionary

    def process_claim(self, claim: Claim, batch_id: str, claim_id: str) -> Claim:
        start_time = time()
        timing_stats = {}

        # Search for relevant papers
        search_start = time()
        relevant_papers = self.literature_searcher.search_papers(claim)
        timing_stats['search_papers'] = time() - search_start

        # Check if no results were found
        if not relevant_papers:
            claim.status = 'processed'
            claim.report = {
                "supportingPapers": [],
                "explanation": "No relevant papers were found for this claim.",
                "claimRating": 0,
                "timing_stats": timing_stats  # Add timing stats to report
            }
            self.update_claim_status(batch_id, claim_id, "processed", json.dumps(claim.report))
            timing_stats['total_time'] = time() - start_time
            return claim

        self.update_claim_status(batch_id, claim_id, "analyzing_papers")
        
        # Process all relevant papers
        processed_papers = []
        non_relevant_papers = []
        inaccessible_papers = []
        total_papers = len(relevant_papers)
        
        paper_processing_start = time()
        paper_timing_stats = []  # Track timing for each paper

        for i, paper in enumerate(relevant_papers):
            paper_start = time()
            paper_stats = {}

            self.update_claim_status(batch_id, claim_id, f"analyzing_paper_{i+1}_of_{total_papers}")
            time_module.sleep(1)
            
            fetch_start = time()
            paper_content, access_info = self.literature_searcher.fetch_paper_content(paper)
            paper_stats['fetch_time'] = time() - fetch_start
            
            if not paper_content:
                inaccessible_papers.append({
                    'paper': paper,
                    'reason': access_info
                })
                continue
            
            analyze_start = time()
            relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages = self.paper_analyzer.analyze_relevance_and_extract(paper_content, claim)
            paper_stats['analyze_time'] = time() - analyze_start
            
            if relevance >= 0.1:
                score_start = time()
                paper_score = self.evidence_scorer.calculate_paper_weight(paper)
                paper_stats['score_time'] = time() - score_start
                
                processed_papers.append({
                    'paper': paper,
                    'relevance': relevance,
                    'excerpts': excerpts,
                    'score': paper_score,
                    'explanations': explanations,
                    'content_type': access_info,
                    'excerpt_pages': excerpt_pages,
                    'processing_time': time() - paper_start,
                    'timing_stats': paper_stats
                })
            else:
                non_relevant_papers.append({
                    'paper': paper,
                    'explanation': non_relevant_explanation or "Paper was determined to be not relevant to the claim.",
                    'relevance': relevance,
                    'score': self.evidence_scorer.calculate_paper_weight(paper),
                    'content_type': access_info,
                    'processing_time': time() - paper_start,
                    'timing_stats': paper_stats
                })

        timing_stats['paper_processing'] = time() - paper_processing_start
        timing_stats['papers'] = paper_timing_stats

        self.update_claim_status(batch_id, claim_id, "generating_report")
        
        # Generate final report
        report_start = time()
        if processed_papers:
            claim.report = self.generate_final_report(claim, processed_papers, non_relevant_papers, inaccessible_papers)
            claim.status = 'processed'
            timing_stats['report_generation'] = time() - report_start
            timing_stats['total_time'] = time() - start_time
            claim.report['timing_stats'] = timing_stats
            self.update_claim_status(batch_id, claim_id, "processed", json.dumps(claim.report))
        else:
            claim.status = 'processed'
            claim.report = {
                "supportingPapers": [],
                "nonRelevantPapers": [
                    {
                        "title": nrp['paper'].title,
                        "content_type": nrp['content_type'],
                        "authors": [
                            {
                                "name": author['name'],
                                "hIndex": self.evidence_scorer.author_h_indices.get(
                                    (author.get('authorId', '')), 0
                                )
                            }
                            for author in nrp['paper'].authors
                        ],
                        "link": nrp['paper'].url,
                        "explanation": nrp['explanation']
                    }
                    for nrp in non_relevant_papers
                ],
                "inaccessiblePapers": [
                    {
                        "title": ip['paper'].title,
                        "authors": [
                            {
                                "name": author['name'],
                                "hIndex": self.evidence_scorer.author_h_indices.get(
                                    (author.get('authorId', '')), 0
                                )
                            }
                            for author in ip['paper'].authors
                        ],
                        "link": ip['paper'].url,
                        "reason": ip['reason']
                    }
                    for ip in inaccessible_papers
                ],
                "explanation": "No relevant papers were found for this claim after analysis.",
                "claimRating": 0,
                "timing_stats": timing_stats
            }
            self.update_claim_status(batch_id, claim_id, "processed", json.dumps(claim.report))
        return claim

    def update_claim_status(self, batch_id: str, claim_id: str, status: str, additional_info: str = "", suggested_claim: str = ""):
        file_path = os.path.join('saved_jobs', batch_id, f"{claim_id}.txt")
        with open(file_path, 'r+') as f:
            content = json.load(f)
            content['status'] = status
            if additional_info:
                content['additional_info'] = additional_info
            if suggested_claim:
                content['suggested_claim'] = suggested_claim
            f.seek(0)
            json.dump(content, f, indent=2)
            f.truncate()

    def process_batch(self, batch_job: BatchJob) -> BatchJob:
        for claim in batch_job.claims:
            self.process_claim(claim, batch_job.id, claim.id)
        batch_job.status = 'processed'
        return batch_job

    def parse_claims(self, input_data: Union[str, pd.DataFrame]) -> List[Claim]:
        if isinstance(input_data, str):
            return [Claim(text=input_data)]
        elif isinstance(input_data, pd.DataFrame):
            return [Claim(text=row['claim']) for _, row in input_data.iterrows()]
        else:
            raise ValueError("Invalid input type. Expected string or DataFrame.")

    def validate_claim_format(self, claim: str) -> dict:
        prompt = f"Evaluate if the following is a valid scientific claim and suggest an optimized version for search:\n\n{claim}"
        system_prompt = dedent("""
        You are an AI assistant tasked with evaluating scientific claims and optimizing them for search. 
        Respond with a JSON object containing 'is_valid' (boolean), 'explanation' (string), and 'suggested' (string).
        The 'is_valid' field should be true if the input is a proper scientific claim, and false otherwise. 
        The 'explanation' field should provide a brief reason for your decision.
        The 'suggested' field should always contain an optimized version of the claim, 
        even if the original claim is invalid. For invalid claims, provide a corrected or improved version.

        Examples of valid scientific claims (note: these may or may not be true, but they are properly formed claims):
         1. "Increased consumption of processed foods is linked to higher rates of obesity in urban populations."
         2. "The presence of certain gut bacteria can influence mood and cognitive function in humans."
         3. "Exposure to blue light from electronic devices before bedtime does not disrupt the circadian rhythm."
         4. "Regular meditation practice can lead to structural changes in the brain's gray matter."
         5. "Higher levels of atmospheric CO2 have no effect on global average temperatures."
         6. "Calcium channels are affected by AMP."
         7. "People who drink soda are much healthier than those who don't."

        Examples of non-claims (these are not valid scientific claims):
         1. "The sky is beautiful." (This is an opinion, not a testable claim)
         2. "What is the effect of exercise on heart health?" (This is a question, not a claim)
         3. "Scientists should study climate change more." (This is a recommendation, not a claim)
         4. "Drink more water!" (This is a command, not a claim),
         5. "Investigating the cognitive effects of BRCA2 mutations on intelligence quotient (IQ) levels." (This doesn't make a claim about anything)

        Reject claims that include ambiguous abbreviations or shorthand, unless it's clear to you what they mean. Remember, a valid scientific claim should be a specific, testable assertion about a phenomenon or relationship between variables. It doesn't have to be true, but it should be a testable assertion.

        For the 'suggested' field, focus on using clear, concise language with relevant scientific terms that would be 
        likely to appear in academic papers. Avoid colloquialisms and ensure the suggested version maintains the 
        original meaning (even if you think it's not true) while being more search-friendly.
        """).strip()
        
        response = self.openai_service.generate_json(prompt, system_prompt=system_prompt)
        return response

    def generate_final_report(self, claim: Claim, processed_papers: List[dict], non_relevant_papers: List[dict], inaccessible_papers: List[dict]) -> dict:
        print("Generating final report")
        # Prepare input for the LLM
        paper_summaries = "\n".join([
            f"Paper: {p['paper'].title}\n"
            f"Authors: {', '.join(author['name'] + ' (H-index: ' + str(self.evidence_scorer.author_h_indices.get(author['authorId'], 0)) + ')' for author in p['paper'].authors)}\n"
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
        Provide an explanation in an essay format with newlines between paragraphs, including specific references to the scientific papers. The essay should have a paragraph highlighting supporting evidence, a paragraph highlighting caveats or contradictions, and then an analysis of which of these outweighs the other and how strongly the claim is supported. Assign a claim rating between -10 (unsupported) and 10 (universally supported).

        The JSON response should have the following structure:
        {
            "explanation": <str containing a short essay explaining the rating>,
            "claimRating": <int between -10 and 10, where 10 is universally supported, 0 is no evidence in either direction, and -10 is universally contradicted>
        }
        """).strip()

        response = self.openai_service.generate_json(prompt, system_prompt)

        # Use the normalize_author_id method from EvidenceScorer
        normalize_author_id = self.evidence_scorer.normalize_author_id

        # Construct the supporting papers data from processed_papers
        supporting_papers = [
            {
                "title": p['paper'].title,
                "authors": [
                    {
                        "name": author['name'],
                        "hIndex": self.evidence_scorer.author_h_indices.get(
                            normalize_author_id(author.get('authorId', '')), 0
                        )
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
                        "page": excerpt_page,  # Assuming excerpt_page is captured during analysis
                        "citation": self.format_citation(p['paper'], excerpt_page)
                    }
                    for excerpt, excerpt_page in zip(p['excerpts'], p.get('excerpt_pages', []))
                ]
            }
            for p in processed_papers
        ]

        # Construct the non-relevant papers data
        non_relevant_papers_data = [
            {
                "title": nrp['paper'].title,
                "content_type": nrp['content_type'],
                "authors": [
                    {
                        "name": author['name'],
                        "hIndex": self.evidence_scorer.author_h_indices.get(
                            normalize_author_id(author.get('authorId', '')), 0
                        )
                    }
                    for author in nrp['paper'].authors
                ],
                "link": nrp['paper'].url,
                "explanation": nrp['explanation']
            }
            for nrp in non_relevant_papers
        ]

        # Construct the inaccessible papers data
        inaccessible_papers_data = [
            {
                "title": ip['paper'].title,
                "content_type": ip['content_type'],
                "authors": [
                    {
                        "name": author['name'],
                        "hIndex": self.evidence_scorer.author_h_indices.get(
                            normalize_author_id(author.get('authorId', '')), 0
                        )
                    }
                    for author in ip['paper'].authors
                ],
                "link": ip['paper'].url,
                "reason": ip['reason']
            }
            for ip in inaccessible_papers
        ]

        # Combine the LLM response with the supporting papers data
        final_report = {
            "supportingPapers": supporting_papers,
            "nonRelevantPapers": non_relevant_papers_data,  # Use the parameter
            "inaccessiblePapers": inaccessible_papers_data,  # Add inaccessible papers
            "explanation": response['explanation'],
            "claimRating": response['claimRating'],
            "searchQueries": self.literature_searcher.saved_search_queries
        }

        # Add usage stats to the final report
        final_report["usage_stats"] = self.openai_service.get_usage_stats()

        return final_report

    def format_citation(self, paper, page_number):
        # Format citation in RIS format for EndNote
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
