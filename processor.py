import asyncio
import os
import json
import logging
from typing import Dict, List, Tuple
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
from app.config import settings
import os.path
import aiofiles
import aiofiles.os as async_os
from contextlib import asynccontextmanager
from filelock import FileLock

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

        # In-memory storage for claims
        self.claims_in_memory: Dict[Tuple[str, str], Dict] = {}

        # Processing status flags
        self.claims_query_generation_in_progress = set()
        self.claims_searching_in_progress = set()
        self.papers_analyzing_in_progress = set()
        self.papers_scoring_in_progress = set()
        self.claims_final_reporting_in_progress = set()

        # Token tracking
        self.claim_token_usage = {}
        self.max_tokens_per_claim = 300000
        self.request_token_estimates = []
        self.max_tokens_per_window = 25000
        self.max_requests_per_window = 10
        self.window_size_seconds = 5
        self.last_token_update_time = time.time()
        self.model = settings.Config.LLM_EVALUATION_MODEL

        self._active_locks = set()

    async def _add_tokens_for_claim(self, claim_id: str, tokens: float, batch_id: str):
        """Track token usage for a claim and handle over-limit cases."""
        current_usage = self.claim_token_usage.get(claim_id, 0)
        new_usage = current_usage + tokens
        self.claim_token_usage[claim_id] = new_usage

        if new_usage > self.max_tokens_per_claim:
            logger.warning(f"Claim {claim_id} exceeded token cap of {self.max_tokens_per_claim}. Marking as processed.")
            claim_data = self.claims_in_memory.get((batch_id, claim_id))
            if claim_data:
                claim_data['status'] = 'processed'
                claim_data['report'] = {
                    "relevantPapers": [],
                    "explanation": "Stopped: token usage exceeded our cap.",
                    "claimRating": 0,
                    "timing_stats": {},
                    "searchQueries": claim_data.get('semantic_scholar_queries', []),
                    "claim_text": claim_data.get('text', '')
                }
                await self._save_processed_claim(claim_data, batch_id, claim_id)

    async def _save_processed_claim(self, claim_data: Dict, batch_id: str, claim_id: str):
        """Save a processed claim to disk and handle cleanup."""
        saved_batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
        os.makedirs(saved_batch_dir, exist_ok=True)
        
        file_path = os.path.join(saved_batch_dir, f"{claim_id}.txt")
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(claim_data, indent=2))
        
        # Remove from memory
        self.claims_in_memory.pop((batch_id, claim_id), None)
        
        # Check if batch is complete
        await self._check_batch_completion(batch_id)

    async def _check_batch_completion(self, batch_id: str):
        """Check if all claims in a batch are processed and handle notifications."""
        batch_claims = [(b, c) for (b, c) in self.claims_in_memory.keys() if b == batch_id]
        if not batch_claims:
            # All claims processed, check for notification
            notification_file = os.path.join(QUEUED_JOBS_DIR, batch_id, 'notification.json')
            if await async_os.path.exists(notification_file):
                async with aiofiles.open(notification_file, 'r') as f:
                    notification_data = json.loads(await f.read())
                if notification_data.get('email'):
                    self.email_service.send_batch_completion_notification(
                        notification_data['email'],
                        batch_id,
                        notification_data.get('num_claims', 0),
                        notification_data.get('review_type', 'standard')
                    )
                # Remove the batch directory
                batch_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
                if await async_os.path.exists(batch_dir):
                    await self.async_rmtree(batch_dir)

    def calculate_tokens_in_window(self):
        """Calculate token usage within the current window."""
        current_time = time.time()
        self.request_token_estimates = [
            estimate for estimate in self.request_token_estimates 
            if current_time - estimate['timestamp'] < self.window_size_seconds
        ]
        num_requests = len(self.request_token_estimates)
        return (num_requests, sum(estimate['tokens'] for estimate in self.request_token_estimates))

    async def generate_search_queries(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Generate search queries for a claim."""
        try:
            # Token tracking
            estimated_tokens = 1000 + (len(claim_data['text']) / 3.5)
            self.request_token_estimates.append({'tokens': estimated_tokens, 'timestamp': time.time()})
            await self._add_tokens_for_claim(claim_id, estimated_tokens, batch_id)

            # Check token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            # Generate queries
            queries, usage = await self.s2_searcher.generate_search_queries(
                claim_data['text'],
                claim_data['search_config']['num_queries'],
                ai_service=self.openai_service
            )

            # Update in-memory claim data
            claim_data = self.claims_in_memory[(batch_id, claim_id)]
            claim_data['semantic_scholar_queries'] = queries
            claim_data['status'] = 'ready_for_search'
            claim_data['usage'] = usage

        except Exception as e:
            logger.error(f"Error generating search queries for claim {claim_id}: {str(e)}")
        finally:
            self.claims_query_generation_in_progress.discard(claim_id)

    async def search_papers(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Search for papers relevant to the claim."""
        try:
            # Token tracking
            estimated_tokens = 100  # Small overhead for query processing
            self.request_token_estimates.append({'tokens': estimated_tokens, 'timestamp': time.time()})
            await self._add_tokens_for_claim(claim_id, estimated_tokens, batch_id)

            # Check token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            # Get in-memory claim data
            claim_data = self.claims_in_memory[(batch_id, claim_id)]
            queries = claim_data['semantic_scholar_queries']

            # Search papers
            raw_papers = await self.s2_searcher.search_papers_for_claim(
                queries,
                results_per_query=claim_data['search_config']['results_per_query']
            )
            
            # Process papers
            papers = []
            for raw_paper in raw_papers:
                try:
                    if raw_paper.get('fields_of_study') is None:
                        raw_paper['fields_of_study'] = []
                    papers.append(raw_paper)
                except Exception as e:
                    logger.error(f"Error converting paper {raw_paper.get('corpusId')}: {str(e)}")
                    continue
            
            # Sort by citation count
            papers.sort(key=lambda p: p.get('citationCount', 0), reverse=True)

            # Update claim data in memory
            if not papers:
                claim_data['status'] = 'processed'
                claim_data['report'] = {
                    "relevantPapers": [],
                    "explanation": "No relevant papers were found for this claim.",
                    "claimRating": -1,
                    "timing_stats": {},
                    "searchQueries": queries,
                    "claim_text": claim_data['text']
                }
                # Save final state since we're done
                await self._save_processed_claim(claim_data, batch_id, claim_id)
            else:
                claim_data['raw_papers'] = papers
                claim_data['status'] = 'ready_for_analysis'

        except Exception as e:
            logger.error(f"Error searching for papers for claim {claim_id}: {str(e)}")
        finally:
            self.claims_searching_in_progress.discard(claim_id)

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

    async def analyze_claim(self, claim_data, batch_id: str, claim_id: str) -> None:
        """Analyze the claim."""
        # Initialize lists if they don't exist
        if 'inaccessible_papers' not in claim_data:
            claim_data['inaccessible_papers'] = []
        if 'processed_papers' not in claim_data:
            claim_data['processed_papers'] = []
        if 'non_relevant_papers' not in claim_data:
            claim_data['non_relevant_papers'] = []

        # Get sets of processed paper IDs
        processed_ids = {p['paper']['corpusId'] for p in claim_data.get('processed_papers', [])}
        non_relevant_ids = {p['paper']['corpusId'] for p in claim_data.get('non_relevant_papers', [])}
        inaccessible_ids = {p['corpusId'] for p in claim_data.get('inaccessible_papers', [])}

        # Analyze raw papers
        for raw_paper in claim_data.get('raw_papers', []):
            try:
                await asyncio.sleep(0.1)
                corpus_id = raw_paper.get('corpusId')
                if not corpus_id:
                    logger.warning(f"Raw paper missing corpus ID: {raw_paper}")
                    continue

                if (corpus_id not in processed_ids and 
                    corpus_id not in non_relevant_ids and 
                    corpus_id not in inaccessible_ids):
                    
                    # Get paper content
                    content_dict = self.s2_searcher.get_paper_content(corpus_id)

                    if content_dict is None or content_dict.get('text') is None:
                        # Add to inaccessible papers
                        claim_data['inaccessible_papers'].append(raw_paper)
                        continue

                    raw_paper['content'] = content_dict['text']
                    raw_paper['content_type'] = content_dict['source']

                    estimated_tokens_for_analysis = 1000 + (len(content_dict['text']) / 3.5)
                    current_num_requests, current_num_tokens = self.calculate_tokens_in_window()
                    
                    if (estimated_tokens_for_analysis + current_num_tokens < self.max_tokens_per_window and 
                        current_num_requests < self.max_requests_per_window and
                        corpus_id not in self.papers_analyzing_in_progress and
                        corpus_id not in processed_ids and
                        corpus_id not in non_relevant_ids):
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

        # Score papers that need scoring
        for paper in claim_data['processed_papers']:
            if paper['score'] == -1:
                print(f"Score is -1 for paper {paper['paper']['corpusId']} in claim {claim_id}")
                estimated_tokens_for_scoring = 500
                current_num_requests, current_num_tokens = self.calculate_tokens_in_window()
                
                if (estimated_tokens_for_scoring + current_num_tokens < self.max_tokens_per_window and 
                    current_num_requests < self.max_requests_per_window and
                    paper['paper']['corpusId'] not in self.papers_scoring_in_progress):
                    self.papers_scoring_in_progress.add(paper['paper']['corpusId'])
                    self.request_token_estimates.append({'tokens': estimated_tokens_for_scoring, 'timestamp': time.time()})
                    print(f"Scoring paper {paper['paper']['corpusId']} in claim {claim_id}")
                    await asyncio.sleep(0.1)
                    asyncio.create_task(self.score_paper(paper, batch_id, claim_id))
                else:
                    return

        # Check completion status
        all_papers_scored = all(paper.get('score', -1) != -1 for paper in claim_data.get('processed_papers', []))
        all_papers_processed = all(
            paper['corpusId'] in processed_ids or 
            paper['corpusId'] in non_relevant_ids or 
            paper['corpusId'] in inaccessible_ids 
            for paper in claim_data.get('raw_papers', []) 
            if paper.get('corpusId')
        )

        if all_papers_scored and all_papers_processed:
            print(f"All papers scored and processed for claim {claim_id}")
            if claim_id not in self.claims_final_reporting_in_progress:
                try:
                    print(f"Checking status for final report for claim {claim_id}")
                    estimated_tokens_for_final_report = 2000 + (
                        sum(len(excerpt) for paper in claim_data.get('processed_papers', [])
                            for excerpt in paper.get('excerpts', []) if isinstance(excerpt, str)) +
                        sum(len(explanation) for paper in claim_data.get('processed_papers', [])
                            for explanation in paper.get('explanations', []) if isinstance(explanation, str))
                    ) / 3.5
                    
                    current_num_requests, current_num_tokens = self.calculate_tokens_in_window()
                    if (estimated_tokens_for_final_report + current_num_tokens < self.max_tokens_per_window and 
                        current_num_requests < self.max_requests_per_window):
                        self.request_token_estimates.append({'tokens': estimated_tokens_for_final_report, 'timestamp': time.time()})
                        self.claims_final_reporting_in_progress.add(claim_id)
                        print(f"Generating final report for claim {claim_id}")
                        asyncio.create_task(self.generate_final_report(batch_id, claim_id))
                    else:
                        print(f"Claim {claim_id} does not have enough tokens for final report generation")
                        return
                except Exception as e:
                    logger.error(f"Error preparing final report: {e}")
                    return
            else:
                print(f"Claim {claim_id} final report already in progress")
                return
        else:
            print(f"Claim {claim_id} is not fully processed, breaking the loop")
            return
    
    async def _write_claim_data(self, claim_data, batch_id, claim_id):
        """Internal method to write claim data asynchronously."""
        file_path = os.path.join(QUEUED_JOBS_DIR, batch_id, f"{claim_id}.txt")
        try:
            logger.warning(
                f"[_write_claim_data] Writing claim data to {file_path}.\n"
                f"processed_papers scores: {[p['score'] for p in claim_data.get('processed_papers', [])]}"
            )
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(claim_data, indent=2))
            logger.warning(
                f"[_write_claim_data] Finished writing claim data to {file_path}.\n"
                f"processed_papers scores: {[p['score'] for p in claim_data.get('processed_papers', [])]}"
            )
        except Exception as e:
            logger.error(f"Error writing claim data to {file_path}: {str(e)}")
            raise

    async def analyze_single_paper(self, raw_paper, claim_text, batch_id: str, claim_id: str) -> None:
        """Analyze a single paper."""
        try:
            self.papers_analyzing_in_progress.add(raw_paper['corpusId'])
            
            # Estimate tokens before analysis
            estimated_tokens_for_analysis = 1000 + (len(raw_paper['content']) / 3.5)
            await self._add_tokens_for_claim(claim_id, estimated_tokens_for_analysis, batch_id)

            # Check if we exceeded the token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            relevance, excerpts, explanations, non_relevant_explanation, excerpt_pages, usage = (
                await self.paper_analyzer.analyze_relevance_and_extract(
                    raw_paper['content'], 
                    claim_text, 
                    ai_service=self.openai_service
                )
            )

            print(f"Analyzed paper {raw_paper['corpusId']}")

            # Get claim data from memory
            claim_data = self.claims_in_memory[(batch_id, claim_id)]
            
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
                    # Add usage to claim data
                    claim_data['usage']['input_tokens'] += usage['input_tokens']
                    claim_data['usage']['output_tokens'] += usage['output_tokens']
                    claim_data['usage']['cost'] += usage['cost']
                else:
                    logger.warning(f"Skipping duplicate paper {raw_paper['corpusId']} for claim {claim_id}")
            else:
                # Check for duplicates in non_relevant_papers
                non_relevant_corpus_ids = {p['paper']['corpusId'] for p in claim_data.get('non_relevant_papers', [])}
                if raw_paper['corpusId'] not in non_relevant_corpus_ids:
                    claim_data['non_relevant_papers'].append({
                        'paper': raw_paper,
                        'explanation': non_relevant_explanation,
                        'content_type': raw_paper['content_type']
                    })

                # Add usage to claim data
                claim_data['usage']['input_tokens'] += usage['input_tokens']
                claim_data['usage']['output_tokens'] += usage['output_tokens']
                claim_data['usage']['cost'] += usage['cost']

        except Exception as e:
            logger.error(f"Error analyzing paper {raw_paper['corpusId']}: {str(e)}")
        finally:
            self.papers_analyzing_in_progress.discard(raw_paper['corpusId'])

    async def score_paper(self, processed_paper, batch_id: str, claim_id: str) -> None:
        """Score a single paper."""
        try:
            paper_id = processed_paper['paper']['corpusId']
            estimated_tokens_for_scoring = 500
            await self._add_tokens_for_claim(claim_id, estimated_tokens_for_scoring, batch_id)

            # Check token cap
            if claim_id in self.claim_token_usage and self.claim_token_usage[claim_id] > self.max_tokens_per_claim:
                return

            score, usage = await self.evidence_scorer.calculate_paper_weight(
                processed_paper, 
                ai_service=self.openai_service
            )

            print(f"Claim {claim_id} token usage as of scoring: {self.claim_token_usage[claim_id]}")

            # Get claim data from memory
            claim_data = self.claims_in_memory[(batch_id, claim_id)]

            # Update score
            for paper in claim_data['processed_papers']:
                if paper['paper']['corpusId'] == paper_id:
                    print(f"Found paper to set score for: {paper_id}")
                    paper['score'] = score
                    break

            # Add usage to claim data
            claim_data['usage']['input_tokens'] += usage['input_tokens']
            claim_data['usage']['output_tokens'] += usage['output_tokens']
            claim_data['usage']['cost'] += usage['cost']

        except Exception as e:
            logger.error(f"Error scoring paper: {str(e)}")
        finally:
            if 'paper_id' in locals():
                self.papers_scoring_in_progress.discard(paper_id)

    async def generate_final_report(self, batch_id: str, claim_id: str) -> None:
        """Generate the final report."""
        try:
            # Get claim data from memory
            claim_data = self.claims_in_memory[(batch_id, claim_id)]
            
            # Estimate tokens for final report generation
            estimated_tokens_for_final_report = 2000 + (
                sum(len(excerpt) for paper in claim_data.get('processed_papers', [])
                    for excerpt in paper.get('excerpts', []) if isinstance(excerpt, str)) +
                sum(len(explanation) for paper in claim_data.get('processed_papers', [])
                    for explanation in paper.get('explanations', []) if isinstance(explanation, str))
            ) / 3.5
            
            await self._add_tokens_for_claim(claim_id, estimated_tokens_for_final_report, batch_id)

            # Check token cap
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
                    "claimRating": -1,
                    "timing_stats": {},
                    "searchQueries": claim_data['semantic_scholar_queries'],
                    "claim_text": claim_data['text']
                }
                claim_data['report'] = report
            else:
                report, usage = await self.claim_processor.generate_final_report(
                    claim_data['text'],
                    claim_data['processed_papers'],
                    claim_data['non_relevant_papers'],
                    claim_data['inaccessible_papers'],
                    claim_data['semantic_scholar_queries'],
                    ai_service=self.openai_service
                )
                
                claim_data['status'] = "processed"
                # Add usage to claim data
                claim_data['usage']['input_tokens'] += usage['input_tokens']
                claim_data['usage']['output_tokens'] += usage['output_tokens']
                claim_data['usage']['cost'] += usage['cost']

                # Add usage to report
                report['usage_stats'] = {
                    'input_tokens': claim_data['usage']['input_tokens'],
                    'output_tokens': claim_data['usage']['output_tokens'],
                    'total_cost': claim_data['usage']['cost']
                }

                claim_data['report'] = report

            # Save the final processed claim
            await self._save_processed_claim(claim_data, batch_id, claim_id)

        except Exception as e:
            logger.error(f"Error preparing final report: {e}")
        finally:
            self.claims_final_reporting_in_progress.discard(claim_id)

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
                        asyncio.sleep(0.1)
                        claim_id = filename[:-4]  # Remove .txt
                        
                        # Skip if already in memory
                        if (batch_id, claim_id) in self.claims_in_memory:
                            claim_data = self.claims_in_memory[(batch_id, claim_id)]
                        else:
                            # Load new claim into memory
                            async with aiofiles.open(file_path, 'r') as f:
                                claim_data = json.loads(await f.read())
                                self.claims_in_memory[(batch_id, claim_id)] = claim_data
                        
                        # Process based on status
                        if claim_data['status'] == 'queued':
                            if claim_id not in self.claims_query_generation_in_progress:
                                estimated_tokens = 1000 + (len(claim_data['text']) / 3.5)
                                current_requests, current_tokens = self.calculate_tokens_in_window()
                                if (estimated_tokens + current_tokens < self.max_tokens_per_window and 
                                    current_requests < self.max_requests_per_window):
                                    self.claims_query_generation_in_progress.add(claim_id)
                                    asyncio.create_task(self.generate_search_queries(claim_data, batch_id, claim_id))
                                
                        elif claim_data['status'] == 'ready_for_search':
                            if not self.claims_searching_in_progress:
                                self.claims_searching_in_progress.add(claim_id)
                                asyncio.create_task(self.search_papers(claim_data, batch_id, claim_id))
                                
                        elif claim_data['status'] == 'ready_for_analysis':
                            current_requests, current_tokens = self.calculate_tokens_in_window()
                            if current_tokens < self.max_tokens_per_window and current_requests < self.max_requests_per_window:
                                await self.analyze_claim(claim_data, batch_id, claim_id)
                                
                        elif claim_data['status'] == 'processed':
                            # Clean up memory and move file
                            self.claims_in_memory.pop((batch_id, claim_id), None)
                            saved_batch_dir = os.path.join(SAVED_JOBS_DIR, batch_id)
                            os.makedirs(saved_batch_dir, exist_ok=True)
                            await self.async_move_file(file_path, os.path.join(saved_batch_dir, f"{claim_id}.txt"))
                            
                            # Check if batch is complete
                            await self._check_batch_completion(batch_id)

                    except Exception as e:
                        logger.error(f"Error processing claim file {file_path}: {str(e)}")

        except Exception as e:
            logger.error(f"Error checking for claims: {str(e)}")

    # Create an async wrapper for shutil operations
    async def async_move_file(self, src, dst):
        await asyncio.to_thread(shutil.move, src, dst)

    async def async_rmtree(self, path):
        await asyncio.to_thread(shutil.rmtree, path)

async def main():
    """Main function to run the processor."""
    try:
        await asyncio.to_thread(os.makedirs, QUEUED_JOBS_DIR, exist_ok=True)
        await asyncio.to_thread(os.makedirs, SAVED_JOBS_DIR, exist_ok=True)

        # Clear all lock files in every subdirectory of QUEUED_JOBS_DIR
        for batch_id in os.listdir(QUEUED_JOBS_DIR):
            batch_dir = os.path.join(QUEUED_JOBS_DIR, batch_id)
            if os.path.isdir(batch_dir):
                for filename in os.listdir(batch_dir):
                    if filename.endswith('.lock'):
                        os.remove(os.path.join(batch_dir, filename))

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
