import asyncio
import os
import json
import logging
from typing import Dict, List
import shutil
from app.services.claim_processor import ClaimProcessor
from semantic_scholar.utils.searcher import S2Searcher
from app.models.claim import Claim
from app.models.paper import Paper
from app.services.email_service import EmailService
from app.services.openai_service import OpenAIService
from app.services.paper_analyzer import PaperAnalyzer
from app.services.evidence_scorer import EvidenceScorer
import time
from filelock import FileLock
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

QUEUED_JOBS_DIR = 'queued_jobs'
SAVED_JOBS_DIR = 'saved_jobs'

class ValsciProcessor:
    def __init__(self):
        self.s2_searcher = S2Searcher()
        self.paper_analyzer = PaperAnalyzer()
        self.evidence_scorer = EvidenceScorer()
        self.claim_processor = ClaimProcessor()
        self.openai_service = OpenAIService()
        self.email_service = EmailService()

        self.claims_query_generation_in_progress = {}
        self.claims_searching_in_progress = {}
        self.papers_analyzing_in_progress = {}
        self.papers_scoring_in_progress = {}
        self.claims_final_reporting_in_progress = {}


        self.request_token_estimates = []
        self.max_tokens_per_minute = 450000
        self.last_token_update_time = time.time()

    def calculate_tokens_in_last_minute(self):
        # Remove any estimates that are older than 1 minute
        self.request_token_estimates = [estimate for estimate in self.request_token_estimates if time.time() - estimate['timestamp'] < 60]
        # Return the sum of the remaining estimates
        return sum(estimate['tokens'] for estimate in self.request_token_estimates)

    async def generate_search_queries(self, claim_data, batch_id: str, claim_id: str) -> List[str]:
        """Generate search queries for a claim."""
        # Estimate the number of tokens for the query generation
        estimated_tokens_for_query_generation = 1000 + (len(claim_data['text']) / 3.5)
        # Add the estimate to the list
        self.request_token_estimates.append({'tokens': estimated_tokens_for_query_generation, 'timestamp': time.time()})
        queries = await self.s2_searcher.generate_search_queries(claim_data['text'], claim_data['search_config']['num_queries'], ai_service=self.openai_service)
        # Save the queries to a file
        # First load the current file
        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
            claim_data = json.load(f)
        # Add the queries to the claim data
        claim_data['semantic_scholar_queries'] = queries
        # Add the status to the claim data
        claim_data['status'] = 'ready_for_search'
        # Save the updated claim data back to the file
        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
            json.dump(claim_data, f, indent=2)

        return
    
    async def search_papers(self, claim_data, batch_id: str, claim_id: str) -> List[Paper]:
        """Search for papers relevant to the claim."""

        queries = claim_data['semantic_scholar_queries']

        raw_papers = await self.s2_searcher.search_papers_for_claim(
            queries,
            results_per_query=claim_data['search_config']['results_per_query']
        )
        
        papers = []
        for raw_paper in raw_papers:
            try:
                # Ensure fields_of_study is a list
                if raw_paper.get('fields_of_study') is None:
                    raw_paper['fields_of_study'] = []
                papers.append(raw_paper)
            except Exception as e:
                logger.error(f"Error converting paper {raw_paper.get('corpusId')}: {str(e)}")
                continue
        
        # Sort by citation count (most cited first)
        papers.sort(key=lambda p: p.citation_count or 0, reverse=True)

        # If no papers are found, set the status to "no_papers_found"
        if not papers:
            claim_data['status'] = 'processed'
            report = {
                    "relevantPapers": [],
                    "explanation": "No relevant papers were found for this claim.",
                    "claimRating": 0,
                    "timing_stats": {},
                    "searchQueries": queries,
                    "claim_text": claim_data['text']
                }
            claim_data['report'] = report
        else:
            claim_data['raw_papers'] = papers
            claim_data['status'] = 'ready_for_analysis'

        # Save the updated claim data back to the file
        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
            json.dump(claim_data, f, indent=2)

        return
    
    def analyze_claim(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Analyze the claim."""
        
        # Use FileLock for initial list creation
        with FileLock(f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"):
            if 'inaccessible_papers' not in claim_data or 'processed_papers' not in claim_data or 'non_relevant_papers' not in claim_data:
                # Make sure the lists exist
                if 'inaccessible_papers' not in claim_data:
                    claim_data['inaccessible_papers'] = []
                if 'processed_papers' not in claim_data:
                    claim_data['processed_papers'] = []
                if 'non_relevant_papers' not in claim_data:
                    claim_data['non_relevant_papers'] = []
                # Save the updated claim data back to the file
                with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
                    json.dump(claim_data, f, indent=2)

        # Iterate over the raw papers and check if any of them aren't in processed_papers or non_relevant_papers
        for raw_paper in claim_data['raw_papers']:
            if raw_paper['corpusId'] not in [paper['paper']['corpusId'] for paper in claim_data['processed_papers']] \
            and raw_paper['corpusId'] not in [paper['paper']['corpusId'] for paper in claim_data['non_relevant_papers']] \
            and raw_paper['corpusId'] not in [paper['paper']['corpusId'] for paper in claim_data['inaccessible_papers']]:
                # Get the content of the paper
                content_dict = self.s2_searcher.get_paper_content(raw_paper['corpusId'])

                if content_dict['text'] is None:
                    # Add to inaccessible papers using FileLock
                    with FileLock(f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"):
                        # First load the current file
                        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                            claim_data = json.load(f)
                        # Add the inaccessible paper to the claim data
                        claim_data['inaccessible_papers'].append(raw_paper)
                        # Save the updated claim data back to the file
                        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
                            json.dump(claim_data, f, indent=2)
                    continue

                raw_paper['content'] = content_dict['text']
                raw_paper['content_type'] = content_dict['source']

                # Check the estimated tokens for the analysis
                estimated_tokens_for_analysis = 1000 + (len(content_dict['text']) / 3.5)
                if estimated_tokens_for_analysis + self.calculate_tokens_in_last_minute() < self.max_tokens_per_minute and not self.papers_analyzing_in_progress.get(raw_paper['corpusId']):
                    # Start the analysis without blocking
                    self.request_token_estimates.append({'tokens': estimated_tokens_for_analysis, 'timestamp': time.time()})
                    self.papers_analyzing_in_progress[raw_paper['corpusId']] = True
                    asyncio.create_task(self.analyze_single_paper(raw_paper, claim_data['text'], batch_id, claim_id))
                else:
                    # Skip the rest of this claim until the next iteration
                    return

        # Use FileLock when checking and updating processed papers
        with FileLock(f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"):
            # Reload claim data to get latest state
            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                claim_data = json.load(f)
            
            # If we have processed papers but with a score of -1, we need to score them
            for paper in claim_data['processed_papers']:
                if paper['score'] == -1:
                    # Check the estimated tokens for the scoring (always 500 for this)
                    estimated_tokens_for_scoring = 500
                    if estimated_tokens_for_scoring + self.calculate_tokens_in_last_minute() < self.max_tokens_per_minute and not self.papers_scoring_in_progress.get(paper['paper']['corpusId']):
                        self.papers_scoring_in_progress[paper['paper']['corpusId']] = True
                        self.request_token_estimates.append({'tokens': estimated_tokens_for_scoring, 'timestamp': time.time()})
                        asyncio.create_task(self.score_paper(paper, claim_data, batch_id, claim_id))
                    else:
                        # Skip the rest of this claim until the next iteration
                        return

            # Check completion status under the same lock
            if all(paper['score'] != -1 for paper in claim_data['processed_papers']) \
            and all(raw_paper['corpusId'] in [paper['paper']['corpusId'] for paper in claim_data['processed_papers']] \
            or raw_paper['corpusId'] in [paper['paper']['corpusId'] for paper in claim_data['non_relevant_papers']] \
            or raw_paper['corpusId'] in [paper['paper']['corpusId'] for paper in claim_data['inaccessible_papers']] \
            for raw_paper in claim_data['raw_papers']):
                if not self.claims_final_reporting_in_progress.get(claim_id):
                    estimated_tokens_for_final_report = 1000 + (sum(len(excerpt) for paper in claim_data['processed_papers'] for excerpt in paper['excerpts']) + sum(len(explanation) for paper in claim_data['processed_papers'] for explanation in paper['explanations'])) / 3.5
                    if estimated_tokens_for_final_report + self.calculate_tokens_in_last_minute() < self.max_tokens_per_minute:
                        self.request_token_estimates.append({'tokens': estimated_tokens_for_final_report, 'timestamp': time.time()})
                        self.claims_final_reporting_in_progress[claim_id] = True
                        # Generate the final report
                        asyncio.create_task(self.generate_final_report(claim_data, batch_id, claim_id))
                    else:
                        # Skip the rest of this claim until the next iteration
                        return

        return
    
    async def analyze_single_paper(self, raw_paper, claim_text, batch_id: str, claim_id: str) -> None:
        """Analyze a single paper."""
        try:
            self.papers_analyzing_in_progress[raw_paper['corpusId']] = True
            relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages = (
                await self.paper_analyzer.analyze_relevance_and_extract(raw_paper['content'], claim_text, ai_service=self.openai_service)
            )

            # Load the latest claim data
            with FileLock(f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"):
                with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                    claim_data = json.load(f)

                if relevance >= 0.1:
                    claim_data['processed_papers'].append({
                        'paper': raw_paper,
                        'relevance': relevance,
                        'excerpts': excerpts,
                        'score': -1,
                        'explanations': explanations,
                        'content_type': raw_paper['content_type'],
                        'excerpt_pages': excerpt_pages
                    })
                else:
                    claim_data['non_relevant_papers'].append({
                        'paper': raw_paper,
                        'explanation': non_relevant_explanation,
                        'content_type': raw_paper['content_type']
                    })

                # Save the updated data
                self.save_claim_data(claim_data, batch_id, claim_id)

        except Exception as e:
            logger.error(f"Error analyzing paper {raw_paper['corpusId']}: {str(e)}")
        finally:
            self.papers_analyzing_in_progress[raw_paper['corpusId']] = False

        return
    
    async def score_paper(self, raw_paper, claim_data, batch_id: str, claim_id: str) -> None:
        """Score a single paper."""
        try:
            self.papers_scoring_in_progress[raw_paper['corpusId']] = True
            score = await self.evidence_scorer.calculate_paper_weight(raw_paper, ai_service=self.openai_service)

            # Load the latest claim data using FileLock
            with FileLock(f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"):
                with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                    claim_data = json.load(f)

                # Update the score in the claim data
                for paper in claim_data['processed_papers']:
                    if paper['paper']['corpusId'] == raw_paper['corpusId']:
                        paper['score'] = score
                        break

                # Save using the safe method
                self.save_claim_data(claim_data, batch_id, claim_id)

        except Exception as e:
            logger.error(f"Error scoring paper {raw_paper['corpusId']}: {str(e)}")
        finally:
            self.papers_scoring_in_progress[raw_paper['corpusId']] = False

        return
    
    async def generate_final_report(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Generate the final report."""

        # Check that the length of processed_papers is greater than 0
        if len(claim_data['processed_papers']) == 0:
            logger.error("No processed papers found for claim {claim_id}")
            # Set the status to "processed"
            claim_data['status'] = "processed"
            report = {
                "relevantPapers": [],
                "nonRelevantPapers": self.claim_processor._format_non_relevant_papers(claim_data['non_relevant_papers']),
                "inaccessiblePapers": self.claim_processor._format_inaccessible_papers(claim_data['inaccessible_papers']),
                "explanation": "No relevant papers were found that support or refute this claim.",
                "claimRating": 0,
                "timing_stats": {},
                "searchQueries": claim_data['semantic_scholar_queries'],
                "claim_text": claim_data['text']
            }
            claim_data['report'] = report
            # Save the updated claim data back to the file
            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
                json.dump(claim_data, f, indent=2)
            return

        else:
            report = await self.claim_processor.generate_final_report(
                claim_data['text'],
                claim_data['processed_papers'],
                claim_data['non_relevant_papers'],
                claim_data['inaccessible_papers'],
                claim_data['semantic_scholar_queries'],
                ai_service=self.openai_service
            )
            claim_data['report'] = report

            # Set the status to "processed"
            claim_data['status'] = "processed"

            # Save the updated claim data back to the file
            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
                json.dump(claim_data, f, indent=2)

        self.claims_final_reporting_in_progress[claim_id] = False
        return


    async def check_for_claims(self):
        """Check for any queued claims and process them."""
        try:
            if not os.path.exists(QUEUED_JOBS_DIR):
                return

            for batch_id in os.listdir(QUEUED_JOBS_DIR):
                batch_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
                if not os.path.isdir(batch_dir):
                    continue

                for filename in os.listdir(batch_dir):
                    if not filename.endswith('.txt') or filename == 'claims.txt':
                        continue

                    file_path = os.path.join(batch_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            claim_data = json.load(f)

                        claim_id = claim_data['claim_id']
                        batch_id = claim_data['batch_id']
                            
                        if claim_data.get('status') == 'queued':
                            if not self.claims_query_generation_in_progress.get(claim_id):
                                # create a query generation task and execute it (not blocking) if the estimated tokens are less than the max
                                # The estimated tokens is 1000 plus (claim_text_length / 3.5)
                                # otherwise, add it to the queue
                                claim_text_length = len(claim_data['text'])
                                estimated_tokens_for_query_generation = 1000 + (claim_text_length / 3.5)
                                if estimated_tokens_for_query_generation + self.calculate_tokens_in_last_minute() < self.max_tokens_per_minute:
                                    self.claims_query_generation_in_progress[claim_id] = True
                                    asyncio.create_task(self.generate_search_queries(claim_data, batch_id, claim_id))
                                else:
                                    # skip this claim until the next iteration
                                    continue
                        if claim_data.get('status') == 'ready_for_search':
                            # Check whether there is currently a claim with searching in progress
                            if any(self.claims_searching_in_progress.get(claim_id) for claim_id in self.claims_searching_in_progress):
                                # skip this claim until the next iteration
                                continue
                            else:
                                # Mark this claim as searching in progress
                                self.claims_searching_in_progress[claim_id] = True
                                # create a search task and execute it (not blocking)
                                asyncio.create_task(self.search_papers(claim_data, batch_id, claim_id))

                        if claim_data.get('status') == 'ready_for_analysis':
                            # Only start if the estimated tokens for the analysis are less than the max
                            if self.calculate_tokens_in_last_minute() < self.max_tokens_per_minute:
                                self.analyze_claim(claim_data, batch_id, claim_id)
                            else:
                                # skip this claim until the next iteration
                                continue         

                        if claim_data.get('status') == 'processed':
                            # Move to saved jobs directory
                            saved_batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
                            os.makedirs(saved_batch_dir, exist_ok=True)
                            shutil.move(file_path, os.path.join(saved_batch_dir, f"{claim_id}.txt"))

                            # Send an email if notifications are enabled and there are no more queued claims in this batch
                            # There is a notification.json file with fields email and num_claims
                            # First check that there are no more queued claims in this batch
                            queued_claims = [filename for filename in os.listdir(batch_dir) if filename.endswith('.txt') and filename != 'claims.txt']
                            if len(queued_claims) == 0:
                                # Then check that there is a notification.json file
                                notification_file = os.path.join(batch_dir, 'notification.json')
                                if os.path.exists(notification_file):
                                    with FileLock(f"{notification_file}.lock"):
                                        with open(notification_file, 'r') as f:
                                            notification_data = json.load(f)
                                        if notification_data['email']:
                                            self.email_service.send_batch_complete_notification(notification_data['email'], batch_id)
                                        # Also remove the batch directory from the queued_jobs directory
                                        shutil.rmtree(batch_dir)

                            logger.info(f"Completed processing claim {claim_id}")

                    except Exception as e:
                        logger.error(f"Error checking claim file {file_path}: {str(e)}")

        except Exception as e:
            logger.error(f"Error checking for queued claims: {str(e)}")

    def save_claim_data(self, claim_data, batch_id, claim_id):
        """Safely save claim data to file using a file lock."""
        file_path = os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt")
        lock_path = f"{file_path}.lock"
        
        with FileLock(lock_path):
            with open(file_path, 'w') as f:
                json.dump(claim_data, f, indent=2)

async def main():
    """Main function to run the processor."""
    try:
        os.makedirs(QUEUED_JOBS_DIR, exist_ok=True)
        os.makedirs(SAVED_JOBS_DIR, exist_ok=True)

        processor = ValsciProcessor()
        logger.info("Started monitoring queued_jobs directory")

        while True:
            try:
                await processor.check_for_claims()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in check_for_claims: {str(e)}")
                # Don't let one error stop the entire process
                continue

    except KeyboardInterrupt:
        logger.info("Shutting down processor")
        # Clean up any in-progress tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
    except Exception as e:
        logger.error(f"Fatal error in main loop: {str(e)}")
        raise
    finally:
        # Additional cleanup if needed
        logger.info("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())
