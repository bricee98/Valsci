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
import time
from textwrap import dedent

class ClaimProcessor:
    def __init__(self):
        self.literature_searcher = LiteratureSearcher()
        self.paper_analyzer = PaperAnalyzer()
        self.evidence_scorer = EvidenceScorer()
        self.openai_service = OpenAIService()

    def process_claim(self, claim: Claim, batch_id: str, claim_id: str) -> Claim:
        # Validate the claim
        validation_result = self.validate_claim_format(claim.text)
        if not validation_result['is_valid']:
            claim.status = 'invalid'
            claim.validation_message = validation_result['explanation']
            self.update_claim_status(batch_id, claim_id, "invalid", claim.validation_message, validation_result['suggested'])
            return claim

        # If the claim is valid, update with the suggested claim
        self.update_claim_status(batch_id, claim_id, "validating", suggested_claim=validation_result['suggested'])
        
        # Search for relevant papers
        relevant_papers = self.literature_searcher.search_papers(claim)

        # Check if no results were found
        if not relevant_papers:
            claim.status = 'processed'
            claim.report = {
                "supportingPapers": [],
                "explanation": "No relevant papers were found for this claim.",
                "claimRating": 0
            }
            self.update_claim_status(batch_id, claim_id, "processed", json.dumps(claim.report))
            return claim

        self.update_claim_status(batch_id, claim_id, "analyzing_papers")
        # Process all relevant papers
        processed_papers = []
        self.non_relevant_papers = []  # Make this an instance variable
        inaccessible_papers = []  # New list for tracking inaccessible papers
        total_papers = len(relevant_papers)
        for i, paper in enumerate(relevant_papers):
            self.update_claim_status(batch_id, claim_id, f"analyzing_paper_{i+1}_of_{total_papers}")
            time.sleep(1)
            paper_content, access_info = self.literature_searcher.fetch_paper_content(paper)
            
            if not paper_content:
                inaccessible_papers.append({
                    'paper': paper,
                    'reason': access_info
                })
                continue
            
            relevance, excerpts, explanations, non_relevant_explanation = self.paper_analyzer.analyze_relevance_and_extract(paper_content, claim)
            if relevance >= 0.1:
                print("Paper is relevant: ", paper.title)
                paper_score = self.evidence_scorer.calculate_paper_weight(paper)
                processed_papers.append({
                    'paper': paper,
                    'relevance': relevance,
                    'excerpts': excerpts,
                    'score': paper_score,
                    'explanations': explanations,
                    'content_type': access_info  # 'full_text' or 'abstract_only'
                })
            else:
                print("Paper is not relevant: ", paper.title)
                self.non_relevant_papers.append({  # Use the instance variable
                    'paper': paper,
                    'explanation': non_relevant_explanation or "Paper was determined to be not relevant to the claim."
                })

        self.update_claim_status(batch_id, claim_id, "generating_report")

        # Generate final report
        if processed_papers:
            claim.report = self.generate_final_report(claim, processed_papers, inaccessible_papers)
            claim.status = 'processed'
            self.update_claim_status(batch_id, claim_id, "processed", json.dumps(claim.report))
        else:
            claim.status = 'processed'
            claim.report = {
                "supportingPapers": [],
                "nonRelevantPapers": self.non_relevant_papers,  # Include non-relevant papers even when no relevant papers found
                "explanation": "No relevant papers were found for this claim after analysis.",
                "claimRating": 0
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
        The 'suggested' field should always contain an optimized version of the claim for searching on Semantic Scholar, 
        even if the original claim is invalid. For invalid claims, provide a corrected or improved version.

        Examples of valid scientific claims (note: these may or may not be true, but they are properly formed claims):
         1. "Increased consumption of processed foods is linked to higher rates of obesity in urban populations."
         2. "The presence of certain gut bacteria can influence mood and cognitive function in humans."
         3. "Exposure to blue light from electronic devices before bedtime disrupts the circadian rhythm."
         4. "Regular meditation practice can lead to structural changes in the brain's gray matter."
         5. "Higher levels of atmospheric CO2 are causing an increase in global average temperatures."
         6. "Calcium channels are affected by AMP."

        Examples of non-claims (these are not valid scientific claims):
         1. "The sky is beautiful." (This is an opinion, not a testable claim)
         2. "What is the effect of exercise on heart health?" (This is a question, not a claim)
         3. "Scientists should study climate change more." (This is a recommendation, not a claim)
         4. "Drink more water!" (This is a command, not a claim)

        Reject claims that include ambiguous abbreviations or shorthand, unless it's clear to you what they mean. Remember, a valid scientific claim should be a specific, testable assertion about a phenomenon or relationship between variables.

        For the 'suggested' field, focus on using clear, concise language with relevant scientific terms that would be 
        likely to appear in academic papers. Avoid colloquialisms and ensure the suggested version maintains the 
        original meaning while being more search-friendly.
        """).strip()
        
        response = self.openai_service.generate_json(prompt, system_prompt=system_prompt)
        return response

    def generate_final_report(self, claim: Claim, processed_papers: List[dict], inaccessible_papers: List[dict]) -> dict:
        print("Generating final report")
        # Prepare input for the LLM
        paper_summaries = "\n".join([
            f"Paper: {p['paper'].title}\n"
            f"Authors: {', '.join(author['name'] + ' (H-index: ' + str(self.evidence_scorer.author_h_indices.get(author['authorId'], 0)) + ')' for author in p['paper'].authors)}\n"
            f"Relevance: {p['relevance']}\nScore: {p['score']}\nExcerpts: {p['excerpts']}"
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

        # Construct the supporting papers data from processed_papers
        supporting_papers = [
            {
                "title": p['paper'].title,
                "authors": [
                    {
                        "name": author['name'],
                        "hIndex": self.evidence_scorer.author_h_indices.get(author['authorId'], 0)
                    }
                    for author in p['paper'].authors
                ],
                "link": p['paper'].url,
                "relevance": p['relevance'],
                "excerpts": p['excerpts'],
                "explanations": p['explanations']
            }
            for p in processed_papers
        ]

        # Combine the LLM response with the supporting papers data
        final_report = {
            "supportingPapers": supporting_papers,
            "nonRelevantPapers": self.non_relevant_papers,  # Use the instance variable
            "explanation": response['explanation'],
            "claimRating": response['claimRating']
        }

        # Add usage stats to the final report
        final_report["usage_stats"] = self.openai_service.get_usage_stats()

        return final_report
