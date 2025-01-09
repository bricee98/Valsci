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
import os.path

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

        # Keep track of token usage per claim
        self.claim_token_usage = {}
        # A per-claim cap on total tokens
        self.max_tokens_per_claim = 300000  # pick your limit

        self.request_token_estimates = []
        self.max_tokens_per_minute = 450000
        self.max_requests_per_minute = 2000
        self.last_token_update_time = time.time()

        self._active_locks = set()

    def _add_tokens_for_claim(self, claim_id: str, tokens: float, batch_id: str):
        """
        Accumulate token usage for a given claim and check for over-limit.
        If over limit, immediately mark the claim as processed.
        """
        current_usage = self.claim_token_usage.get(claim_id, 0)
        new_usage = current_usage + tokens
        self.claim_token_usage[claim_id] = new_usage

        if new_usage > self.max_tokens_per_claim:
            # Mark claim as processed to avoid infinite loops
            logger.warning(f"Claim {claim_id} exceeded token cap of {self.max_tokens_per_claim}. Marking as processed.")
            lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
            with FileLock(lock_path):
                # Load current file
                file_path = os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        print(f"Loading claim data for max tokens cap in claim {claim_id}")
                        claim_data = json.load(f)

                    claim_data['status'] = 'processed'
                    claim_data['report'] = {
                        "relevantPapers": [],
                        "explanation": "Stopped: token usage exceeded our cap.",
                        "claimRating": 0,
                        "timing_stats": {},
                        "searchQueries": claim_data.get('semantic_scholar_queries', []),
                        "claim_text": claim_data.get('text', '')
                    }

                    # Write updated data to file
                    with open(file_path, 'w') as f:
                        json.dump(claim_data, f, indent=2)

    def calculate_tokens_in_last_minute(self):
        # Remove any estimates that are older than 1 minute
        self.request_token_estimates = [estimate for estimate in self.request_token_estimates if time.time() - estimate['timestamp'] < 60]
        # Return the sum of the remaining estimates
        num_requests = len(self.request_token_estimates)
        return (num_requests, sum(estimate['tokens'] for estimate in self.request_token_estimates))

    async def generate_search_queries(self, claim_data, batch_id: str, claim_id: str) -> List[str]:
        """Generate search queries for a claim."""
        estimated_tokens_for_query_generation = 1000 + (len(claim_data['text']) / 3.5)
        self.request_token_estimates.append({'tokens': estimated_tokens_for_query_generation, 'timestamp': time.time()})
        # Also track usage per claim
        self._add_tokens_for_claim(claim_id, estimated_tokens_for_query_generation, batch_id)

        queries = await self.s2_searcher.generate_search_queries(
            claim_data['text'],
            claim_data['search_config']['num_queries'],
            ai_service=self.openai_service
        )

        # If the claim was just marked processed due to exceeding usage, do not proceed
        if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
            return

        # Save the queries to a file
        # First load the current file
        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
            print(f"Generate search queries: Loading claim data for query generation in claim {claim_id}")
            claim_data = json.load(f)
        # Add the queries to the claim data
        claim_data['semantic_scholar_queries'] = queries
        # Add the status to the claim data
        claim_data['status'] = 'ready_for_search'
        # Save the updated claim data back to the file
        with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
            print(f"Generate search queries: Writing claim data for claim {claim_id} in batch {batch_id}")
            json.dump(claim_data, f, indent=2)

        return
    
    async def search_papers(self, claim_data, batch_id: str, claim_id: str) -> List[Paper]:
        """Search for papers relevant to the claim."""
        try:
            # Add a small token estimate for query processing
            estimated_tokens = 100  # Small overhead for query processing
            self.request_token_estimates.append({'tokens': estimated_tokens, 'timestamp': time.time()})
            self._add_tokens_for_claim(claim_id, estimated_tokens, batch_id)

            # Check if we exceeded the token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            queries = claim_data['semantic_scholar_queries']

            raw_papers = await self.s2_searcher.search_papers_for_claim(
                queries,
                results_per_query=claim_data['search_config']['results_per_query']
            )
            
            papers = []
            for raw_paper in raw_papers:
                try:
                    if raw_paper.get('fields_of_study') is None:
                        raw_paper['fields_of_study'] = []
                    papers.append(raw_paper)
                except Exception as e:
                    logger.error(f"Error converting paper {raw_paper.get('corpusId')}: {str(e)}")
                    continue
            
            # Sort by citation count (most cited first)
            papers.sort(key=lambda p: p.get('citationCount', 0), reverse=True)

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

            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
                print(f"Search papers: Writing claim data for claim {claim_id} in batch {batch_id}")
                json.dump(claim_data, f, indent=2)

            return
        
        except Exception as e:
            logger.error(f"Error searching for papers for claim {claim_id}: {str(e)}")

        finally:
            # Reset the searching flag so new searches can proceed
            self.claims_searching_in_progress[claim_id] = False

    def _log_lock(self, action: str, lock_path: str, context: str):
        """Helper to standardize lock logging with context"""
        current_time = time.time()
        if action == "creating":
            self._lock_start_times = getattr(self, '_lock_start_times', {})
            self._lock_start_times[lock_path] = current_time
        elif action == "released":
            start_time = self._lock_start_times.get(lock_path)
            if start_time:
                duration = current_time - start_time
                if duration > 1.0:  # Log warning for locks held more than 1 second
                    logger.warning(f"Lock held for {duration:.2f}s: {lock_path} ({context})")
                del self._lock_start_times[lock_path]
        print(f"Lock {action}: {lock_path} ({context})")

    def analyze_claim(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Analyze the claim."""
        lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
        self._log_lock("creating", lock_path, "initialize claim lists")
        with FileLock(lock_path):
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
                    print(f"Analyze claim: Writing claim data for claim {claim_id} in batch {batch_id}")
                    json.dump(claim_data, f, indent=2)

        # Safely get corpus IDs from processed papers
        processed_ids = []
        for paper in claim_data.get('processed_papers', []):
            try:
                if isinstance(paper, dict) and isinstance(paper.get('paper'), dict):
                    corpus_id = paper['paper'].get('corpusId')
                    if corpus_id:
                        processed_ids.append(corpus_id)
            except Exception as e:
                logger.warning(f"Error getting corpus ID from processed paper: {e}")

        # Safely get corpus IDs from non-relevant papers
        non_relevant_ids = []
        for paper in claim_data.get('non_relevant_papers', []):
            try:
                if isinstance(paper, dict) and isinstance(paper.get('paper'), dict):
                    corpus_id = paper['paper'].get('corpusId')
                    if corpus_id:
                        non_relevant_ids.append(corpus_id)
            except Exception as e:
                logger.warning(f"Error getting corpus ID from non-relevant paper: {e}")

        # Safely get corpus IDs from inaccessible papers
        inaccessible_ids = []
        for paper in claim_data.get('inaccessible_papers', []):
            try:
                corpus_id = paper.get('corpusId')
                if corpus_id:
                    inaccessible_ids.append(corpus_id)
            except Exception as e:
                logger.warning(f"Error getting corpus ID from inaccessible paper: {e}")

        # Iterate over the raw papers with safer checks
        for raw_paper in claim_data.get('raw_papers', []):
            try:
                corpus_id = raw_paper.get('corpusId')
                if not corpus_id:
                    logger.warning(f"Raw paper missing corpus ID: {raw_paper}")
                    continue

                if (corpus_id not in processed_ids and 
                    corpus_id not in non_relevant_ids and 
                    corpus_id not in inaccessible_ids):
                    
                    # Get the content of the paper
                    content_dict = self.s2_searcher.get_paper_content(corpus_id)

                    if content_dict is None or content_dict.get('text') is None:
                        # Add to inaccessible papers using FileLock
                        with FileLock(f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"):
                            # First load the current file
                            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                                print(f"Loading claim data for inaccessible paper {raw_paper['corpusId']} in claim {claim_id}")
                                claim_data = json.load(f)
                            # Add the inaccessible paper to the claim data
                            claim_data['inaccessible_papers'].append(raw_paper)
                            # Save the updated claim data back to the file
                            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'w') as f:
                                print(f"Saving updated claim data for inaccessible paper {raw_paper['corpusId']} in claim {claim_id}")
                                json.dump(claim_data, f, indent=2)
                        continue

                    raw_paper['content'] = content_dict['text']
                    raw_paper['content_type'] = content_dict['source']

                    
                    estimated_tokens_for_analysis = 1000 + (len(content_dict['text']) / 3.5)
                    current_num_requests, current_num_tokens = self.calculate_tokens_in_last_minute()
                    if (estimated_tokens_for_analysis + current_num_tokens < self.max_tokens_per_minute and 
                        current_num_requests < self.max_requests_per_minute and
                        not self.papers_analyzing_in_progress.get(corpus_id)):
                        self.request_token_estimates.append({
                            'tokens': estimated_tokens_for_analysis, 
                            'timestamp': time.time()
                        })
                        asyncio.create_task(self.analyze_single_paper(raw_paper, claim_data['text'], batch_id, claim_id))
                    else:
                        return

            except Exception as e:
                logger.error(f"Error processing raw paper: {e}")
                continue

        # Use FileLock when checking and updating processed papers
        lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
        self._log_lock("creating", lock_path, "check and update processed papers")
        with FileLock(lock_path):
            # Reload claim data to get latest state
            with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                print(f"Analyze claim: Loading claim data for processed papers in claim {claim_id}")
                claim_data = json.load(f)
            
            # If we have processed papers but with a score of -1, we need to score them
            for paper in claim_data['processed_papers']:
                if paper['score'] == -1:
                    print(f"Score is -1 for paper {paper['paper']['corpusId']} in claim {claim_id}")
                    # Check the estimated tokens for the scoring (always 500 for this)
                    estimated_tokens_for_scoring = 500
                    current_num_requests, current_num_tokens = self.calculate_tokens_in_last_minute()
                    if (estimated_tokens_for_scoring + current_num_tokens < self.max_tokens_per_minute and 
                        current_num_requests < self.max_requests_per_minute and
                        not self.papers_scoring_in_progress.get(paper['paper']['corpusId'])):
                        self.papers_scoring_in_progress[paper['paper']['corpusId']] = True
                        self.request_token_estimates.append({'tokens': estimated_tokens_for_scoring, 'timestamp': time.time()})
                        asyncio.create_task(self.score_paper(paper, claim_data, batch_id, claim_id))
                    else:
                        # Skip the rest of this claim until the next iteration
                        return

            # Check completion status under the same lock
            all_papers_scored = True
            for paper in claim_data.get('processed_papers', []):
                try:
                    if paper.get('score', -1) == -1:
                        all_papers_scored = False
                        break
                except Exception as e:
                    logger.warning(f"Error checking paper score: {e}")
                    all_papers_scored = False
                    break

            all_papers_processed = True
            try:
                for raw_paper in claim_data.get('raw_papers', []):
                    corpus_id = raw_paper.get('corpusId')
                    if not corpus_id:
                        continue
                        
                    if (corpus_id not in processed_ids and 
                        corpus_id not in non_relevant_ids and 
                        corpus_id not in inaccessible_ids):
                        all_papers_processed = False
                        break
            except Exception as e:
                logger.warning(f"Error checking paper processing status: {e}")
                all_papers_processed = False

            if all_papers_scored and all_papers_processed:
                if not self.claims_final_reporting_in_progress.get(claim_id):
                    try:
                        estimated_tokens_for_final_report = 1000 + (
                            sum(len(excerpt) for paper in claim_data.get('processed_papers', [])
                                for excerpt in paper.get('excerpts', []) if isinstance(excerpt, str)) +
                            sum(len(explanation) for paper in claim_data.get('processed_papers', [])
                                for explanation in paper.get('explanations', []) if isinstance(explanation, str))
                        ) / 3.5
                        current_num_requests, current_num_tokens = self.calculate_tokens_in_last_minute()
                        if (estimated_tokens_for_final_report + current_num_tokens < self.max_tokens_per_minute and 
                            current_num_requests < self.max_requests_per_minute and
                            not self.claims_final_reporting_in_progress.get(claim_id)):
                            self.request_token_estimates.append({'tokens': estimated_tokens_for_final_report, 'timestamp': time.time()})
                            self.claims_final_reporting_in_progress[claim_id] = True
                            # Generate the final report
                            asyncio.create_task(self.generate_final_report(claim_data, batch_id, claim_id))
                        else:
                            # Skip the rest of this claim until the next iteration
                            return
                    except Exception as e:
                        logger.error(f"Error preparing final report: {e}")
                        return

        self._log_lock("released", lock_path, "check and update processed papers")

        return
    
    def _write_claim_data(self, claim_data, batch_id, claim_id):
        """Internal method to write claim data without locking."""
        file_path = os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt")
        logger.warning(
            f"[_write_claim_data] Writing claim data to {file_path}.\n"
            f"processed_papers scores: {[p['score'] for p in claim_data.get('processed_papers', [])]}"
        )
        with open(file_path, 'w') as f:
            print(f"Writing claim data for claim {claim_id} in batch {batch_id}")
            json.dump(claim_data, f, indent=2)
        logger.warning(
            f"[_write_claim_data] Finished writing claim data to {file_path}.\n"
            f"processed_papers scores: {[p['score'] for p in claim_data.get('processed_papers', [])]}"
        )

    async def analyze_single_paper(self, raw_paper, claim_text, batch_id: str, claim_id: str) -> None:
        """Analyze a single paper."""
        try:
            self.papers_analyzing_in_progress[raw_paper['corpusId']] = True
            
            # Estimate tokens before analysis
            estimated_tokens_for_analysis = 1000 + (len(raw_paper['content']) / 3.5)
            self._add_tokens_for_claim(claim_id, estimated_tokens_for_analysis, batch_id)

            # Check if we exceeded the token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages = (
                await self.paper_analyzer.analyze_relevance_and_extract(
                    raw_paper['content'], 
                    claim_text, 
                    ai_service=self.openai_service
                )
            )

            print(f"Analyzed paper {raw_paper['corpusId']}")

            # Single lock context for all file operations
            lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
            self._log_lock("creating", lock_path, f"save analysis for paper {raw_paper['corpusId']}")
            with FileLock(lock_path):
                # Load current state
                with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                    print(f"Loading claim data for analysis of paper {raw_paper['corpusId']} in claim {claim_id}")
                    claim_data = json.load(f)

                # Check for duplicates in processed_papers before adding
                processed_corpus_ids = {p['paper']['corpusId'] for p in claim_data.get('processed_papers', [])}
                
                # Update claim data
                if relevance >= 0.1:
                    if raw_paper['corpusId'] not in processed_corpus_ids:
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
                        logger.warning(f"Skipping duplicate paper {raw_paper['corpusId']} for claim {claim_id}")
                else:
                    # Check for duplicates in non_relevant_papers as well
                    non_relevant_corpus_ids = {p['paper']['corpusId'] for p in claim_data.get('non_relevant_papers', [])}
                    if raw_paper['corpusId'] not in non_relevant_corpus_ids:
                        claim_data['non_relevant_papers'].append({
                            'paper': raw_paper,
                            'explanation': non_relevant_explanation,
                            'content_type': raw_paper['content_type']
                        })

                # Write updated data
                print(f"Analyze claim: Writing claim data for claim {claim_id} in batch {batch_id}")
                self._write_claim_data(claim_data, batch_id, claim_id)
            self._log_lock("released", lock_path, f"save analysis for paper {raw_paper['corpusId']}")

        except Exception as e:
            logger.error(f"Error analyzing paper {raw_paper['corpusId']}: {str(e)}")
        finally:
            self.papers_analyzing_in_progress[raw_paper['corpusId']] = False

    async def score_paper(self, processed_paper, claim_data, batch_id: str, claim_id: str) -> None:
        """Score a single paper."""
        try:
            paper_id = processed_paper['paper']['corpusId']
            # If you want to estimate tokens for scoring:
            estimated_tokens_for_scoring = 500  # as an example
            self._add_tokens_for_claim(claim_id, estimated_tokens_for_scoring, batch_id)
            self.papers_scoring_in_progress[paper_id] = True
            score = await self.evidence_scorer.calculate_paper_weight(
                processed_paper, 
                ai_service=self.openai_service
            )
            # Check if we exceeded the token cap
            print(f"Claim {claim_id} token usage as of scoring: {self.claim_token_usage[claim_id]}")
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            # Single lock context for all file operations
            lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
            self._log_lock("creating", lock_path, f"save score for paper {paper_id}")
            with FileLock(lock_path):
                # Load current state
                with open(os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt"), 'r') as f:
                    print(f"Loading claim data for scoring paper {paper_id} in claim {claim_id}")
                    claim_data = json.load(f)

                # Update score
                for paper in claim_data['processed_papers']:
                    if paper['paper']['corpusId'] == paper_id:
                        print(f"Found paper to set score for: {paper_id}")
                        paper['score'] = score
                        break

                # Write updated data
                print(f"Save score: Writing claim data for claim {claim_id} in batch {batch_id}")
                self._write_claim_data(claim_data, batch_id, claim_id)
                print(f"Updated claim data for scoring paper {paper_id} in claim {claim_id}")
            self._log_lock("released", lock_path, f"save score for paper {paper_id}")

        except Exception as e:
            logger.error(f"Error scoring paper: {str(e)}")
        finally:
            if 'paper_id' in locals():
                self.papers_scoring_in_progress[paper_id] = False

    async def generate_final_report(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Generate the final report."""
        try:
            # Estimate tokens for final report generation
            estimated_tokens_for_final_report = 1000 + (
                sum(len(excerpt) for paper in claim_data.get('processed_papers', [])
                    for excerpt in paper.get('excerpts', []) if isinstance(excerpt, str)) +
                sum(len(explanation) for paper in claim_data.get('processed_papers', [])
                    for explanation in paper.get('explanations', []) if isinstance(explanation, str))
            ) / 3.5
            
            self._add_tokens_for_claim(claim_id, estimated_tokens_for_final_report, batch_id)

            # Check if we exceeded the token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            if len(claim_data['processed_papers']) == 0:
                logger.error(f"No processed papers found for claim {claim_id}")
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

                # Single lock context for file operations
                lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
                self._log_lock("creating", lock_path, f"save empty report for claim {claim_id}")
                with FileLock(lock_path):
                    print(f"Generate final report: Writing claim data for claim {claim_id} in batch {batch_id}")
                    self._write_claim_data(claim_data, batch_id, claim_id)
                self._log_lock("released", lock_path, f"save empty report for claim {claim_id}")
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
                claim_data['status'] = "processed"

                # Single lock context for file operations
                lock_path = f"{os.path.join(QUEUED_JOBS_DIR, batch_id, f'{claim_id}.txt')}.lock"
                self._log_lock("creating", lock_path, f"save final report for claim {claim_id}")
                with FileLock(lock_path):
                    print(f"Generate final report: Writing claim data for claim {claim_id} in batch {batch_id}")
                    self._write_claim_data(claim_data, batch_id, claim_id)
                self._log_lock("released", lock_path, f"save final report for claim {claim_id}")

            self.claims_final_reporting_in_progress[claim_id] = False

        except Exception as e:
            logger.error(f"Error preparing final report: {e}")
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
                            print(f"Loading claim data for check_for_claims in claim {filename}")
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
                                current_num_requests, current_num_tokens = self.calculate_tokens_in_last_minute()
                                if (estimated_tokens_for_query_generation + current_num_tokens < self.max_tokens_per_minute and 
                                    current_num_requests < self.max_requests_per_minute):
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
                            num_requests, num_tokens = self.calculate_tokens_in_last_minute()
                            if num_tokens < self.max_tokens_per_minute and num_requests < self.max_requests_per_minute:
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
                                            print(f"Loading notification data for batch")
                                            notification_data = json.load(f)
                                        if notification_data['email']:
                                            self.email_service.send_batch_completion_notification(notification_data['email'], batch_id, 
                                                                                                notification_data.get('num_claims', 0),
                                                                                                notification_data.get('review_type', 'standard'))
                                        # Also remove the batch directory from the queued_jobs directory
                                        shutil.rmtree(batch_dir)

                            logger.info(f"Completed processing claim {claim_id}")

                    except Exception as e:
                        logger.error(f"Error checking claim file {file_path}: {str(e)}")

        except Exception as e:
            logger.error(f"Error checking for queued claims: {str(e)}")

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
