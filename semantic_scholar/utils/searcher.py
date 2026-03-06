import json
import requests
from pathlib import Path
from typing import Any, List, Dict, Optional, Generator
import time
from rich.console import Console
import ijson
from openai import OpenAI
import asyncio
import mmap
import logging
from app.services.llm.gateway import LLMTask
from app.services.prompt_store import load_prompt, render_prompt
from app.services.llm.validators import OutputValidationError, validate_query_generation_payload

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

project_root = str(Path(__file__).parent.parent.parent)
from app.config.settings import Config

# Import the BinaryIndexer so we can do direct lookups from the .idx files
from semantic_scholar.utils.binary_indexer import BinaryIndexer

console = Console()

class S2Searcher:
    def __init__(self):
        logger.info(f"Initializing S2Searcher with project root: {project_root}")
        self.api_key = Config.SEMANTIC_SCHOLAR_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': self.api_key
        })
        self.base_dir = Path(project_root) / "semantic_scholar/datasets"
        logger.info(f"Base directory set to: {self.base_dir}")
        
        self.rate_limiter = RateLimiter(requests_per_second=1.0)
        
        # Find latest release
        self.current_release = self._get_latest_local_release()
        logger.info(f"Found latest release: {self.current_release}")
        
        # Check for index files
        if self.current_release:
            index_dir = self.base_dir / "binary_indices"
            logger.info(f"Checking index directory: {index_dir}")
            if index_dir.exists():
                index_files = list(index_dir.glob(f"{self.current_release}_*_corpus_id.idx"))
                logger.info(f"Found index files: {[f.name for f in index_files]}")
            else:
                logger.warning(f"Index directory not found: {index_dir}")
        
        self.has_local_data = self.current_release is not None
        logger.info(f"Has local data: {self.has_local_data}")
        if not self.has_local_data:
            logger.warning("No local datasets found. Running in API-only mode.")

        # Now we hold a BinaryIndexer for local lookups:
        self.indexer = BinaryIndexer(self.base_dir)
        logger.info(f"Initialized BinaryIndexer with base dir: {self.base_dir}")

    def _get_latest_local_release(self) -> Optional[str]:
        """Get the latest release from local datasets."""
        logger.info(f"Looking for releases in: {self.base_dir}")
        
        # First check binary_indices directory for metadata
        binary_indices_dir = self.base_dir / "binary_indices"
        logger.info(f"Checking binary indices directory: {binary_indices_dir}")
        
        if binary_indices_dir.exists():
            # Look for metadata files like "2024-12-10_metadata.json"
            metadata_files = list(binary_indices_dir.glob("*_metadata.json"))
            if metadata_files:
                # Extract release IDs from metadata filenames
                releases = [f.name.split('_')[0] for f in metadata_files]
                latest = max(releases) if releases else None
                logger.info(f"Found releases in binary_indices: {releases}, using latest: {latest}")
                return latest
        
        # Fallback to checking main directory
        if not self.base_dir.exists():
            logger.warning(f"Base directory does not exist: {self.base_dir}")
            return None
        
        releases = [d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name == 'binary_indices']
        logger.info(f"Found releases in base directory: {releases}")
        return max(releases) if releases else None

    async def generate_search_queries(
        self,
        claim_text: str,
        num_queries: int = 5,
        ai_service=None,
        batch_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> List[str]:
        """Generate search queries for a claim using GPT."""
        system_prompt = load_prompt("query_generation_system")
        user_prompt = render_prompt(
            "query_generation_user",
            num_queries=num_queries,
            claim_text=claim_text,
        )
        
        try:
            print("About to generate queries")

            result = await ai_service.chat_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                task=LLMTask.QUERY_GENERATION,
                batch_id=batch_id,
                claim_id=claim_id,
                model_override=model_override,
            )
            response = validate_query_generation_payload(
                result['content'],
                expected_query_count=num_queries,
            )
            usage = result['usage']
            queries = response.get('queries', [])
            
            # Log generated queries for debugging
            console.print("\n[cyan]Generated queries:[/cyan]")
            for query in queries:
                console.print(f"[green]- {query}[/green]")
                
            return queries, usage

        except OutputValidationError as e:
            logger.error(f"Query generation output validation failed: {str(e)}")
            if ai_service and hasattr(ai_service, "add_issue"):
                await ai_service.add_issue(
                    batch_id=batch_id,
                    claim_id=claim_id,
                    severity="ERROR",
                    stage=LLMTask.QUERY_GENERATION,
                    message="Query generation output failed schema validation.",
                    details={"error": str(e)},
                )
            raise
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            raise

    async def search_papers_for_claim(self, queries: List[str], results_per_query: int = 5) -> List[Dict]:
        """Search papers relevant to a claim."""
        papers = []
        seen_paper_ids = set()
        
        for query in queries:
            try:
                search_results = await self.search_papers(query, limit=results_per_query)
                
                for paper in search_results:
                    corpus_id = paper.get('corpusId')
                    if not corpus_id:  # Skip papers without corpus ID
                        continue
                        
                    if corpus_id not in seen_paper_ids:
                        console.print(f"[green]Processing new paper with Corpus ID: {corpus_id}[/green]")
                        # Get full content
                        content = self.get_paper_content(corpus_id)
                        if content:  # Only add papers that have content
                            paper['content_source'] = content['source']
                            paper['pdf_hash'] = content['pdf_hash']
                            paper['authors'] = self._enrich_author_data(paper['authors'])
                            seen_paper_ids.add(corpus_id)
                            papers.append(paper)
                    else:
                        console.print(f"[yellow]Skipping duplicate paper with Corpus ID: {corpus_id}[/yellow]")
                        
            except Exception as e:
                console.print(f"[red]Error in search_papers_for_claim: {str(e)}[/red]")
                continue

        papers = [paper for paper in papers if paper is not None]
        return papers

    async def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search papers using S2 API and cross-reference with local data."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                console.print(f"[green]Searching for papers with query: {query}[/green] (Attempt {retry_count + 1}/{max_retries})")
                
                await asyncio.sleep(0.5)

                response = self.session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": limit,
                        "fields": ",".join([
                            'paperId',
                            'corpusId',
                            'title',
                            'abstract',
                            'year',
                            'authors',
                            'venue',
                            'url',
                            'isOpenAccess',
                            'fieldsOfStudy',
                            'citationCount'
                        ])
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get('data'):
                    console.print(f"[yellow]No results found for query: {query}[/yellow]")
                    return []
                
                return data.get('data', [])
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    console.print(f"[yellow]Attempt {retry_count} failed. Retrying...[/yellow]")
                    await asyncio.sleep(1)  # Wait 1 second before retrying
                else:
                    console.print(f"[red]Error in search_papers after {max_retries} attempts: {str(e)}[/red]")
                    return []

    def _get_citation_count(self, paper_id: str) -> int:
        """Get citation count for a paper."""
        citations_data = self._find_in_dataset('citations', paper_id)
        return len(citations_data.get('citations', [])) if citations_data else 0

    def _enrich_author_data(self, authors: List[Dict]) -> List[Dict]:
        """Add additional author information from local dataset for first and last authors only."""
        if not authors:
            return []
        
        # Only keep first and last authors
        key_authors = []
        if len(authors) >= 1:
            key_authors.append(authors[0])  # First author
        if len(authors) >= 2 and authors[-1] != authors[0]:  # Add last author if different from first
            key_authors.append(authors[-1])
        
        enriched_authors = []
        for author in key_authors:
            author_id = author.get('authorId')
            if author_id:
                # Use the binary indexer to look up the author
                local_data = self.indexer.lookup(
                    release_id=self.current_release,
                    dataset='authors',
                    id_type='author_id',
                    search_id=str(author_id)
                )
                if local_data:
                    # Map fields using correct field names from authors dataset
                    author['hIndex'] = local_data.get('hindex', 0)
                    author['paperCount'] = local_data.get('papercount', 0)
                    author['citationCount'] = local_data.get('citationcount', 0)
                    logger.info(f"Enriched author {author_id} with h-index: {author['hIndex']}")
                else:
                    logger.warning(f"No local data found for author: {author_id}")
            enriched_authors.append(author)
        return enriched_authors

    def _build_inaccessible_result(
        self,
        *,
        corpus_id: str,
        reason_code: str,
        reason: str,
        attempts: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            'text': None,
            'source': None,
            'pdf_hash': None,
            'status': 'inaccessible',
            'reason_code': reason_code,
            'reason': reason,
            'lookup_details': {
                'corpus_id': str(corpus_id),
                'release_id': self.current_release,
                'has_local_data': bool(self.has_local_data),
                'attempts': attempts,
            },
        }
        if error:
            result['lookup_details']['error'] = error
        return result

    @staticmethod
    def _attempt_result(dataset: str, status: str, detail: Optional[str] = None) -> Dict[str, str]:
        result = {'dataset': dataset, 'status': status}
        if detail:
            result['detail'] = detail
        return result

    def get_paper_content(self, corpus_id: str) -> Dict[str, Any]:
        """Get full paper content from S2ORC or abstract data, using the BinaryIndexer."""
        logger.info(f"Attempting to get content for corpus ID: {corpus_id}")
        logger.info(f"Current release: {self.current_release}")
        logger.info(f"Has local data: {self.has_local_data}")
        attempts: List[Dict[str, Any]] = []

        if not self.current_release:
            logger.warning("No release ID available")
            return self._build_inaccessible_result(
                corpus_id=str(corpus_id),
                reason_code='missing_release',
                reason='No local dataset release is available for content lookup.',
                attempts=attempts,
            )
        
        try:
            # Try S2ORC first for full text
            logger.info("Attempting S2ORC lookup...")
            s2orc_record = self.indexer.lookup(
                release_id=self.current_release,
                dataset='s2orc',
                id_type='corpus_id',
                search_id=str(corpus_id)
            )
            
            if s2orc_record:
                logger.info("Found record in S2ORC")
                if s2orc_record.get('pdf_parse', {}).get('body_text'):
                    logger.info("Found full text in S2ORC record")
                    full_text = "\n\n".join(
                        section.get('text', '')
                        for section in s2orc_record['pdf_parse']['body_text']
                    )
                    return {
                        'text': full_text,
                        'source': 's2orc',
                        'pdf_hash': s2orc_record.get('pdf_parse', {}).get('pdf_hash'),
                        'status': 'ok',
                        'lookup_details': {
                            'corpus_id': str(corpus_id),
                            'release_id': self.current_release,
                            'has_local_data': bool(self.has_local_data),
                            'attempts': attempts + [self._attempt_result('s2orc', 'found_text')],
                        },
                    }
                else:
                    logger.info("S2ORC record found but no body text available")
                    attempts.append(self._attempt_result('s2orc', 'record_without_text', 'pdf_parse.body_text missing'))
            else:
                attempts.append(self._attempt_result('s2orc', 'missing_record'))


            # Fallback to abstracts dataset
            logger.info("Attempting abstracts lookup...")
            abstract_record = self.indexer.lookup(
                release_id=self.current_release,
                dataset='abstracts',
                id_type='corpus_id',
                search_id=str(corpus_id)
            )
            
            if abstract_record:
                logger.info("Found record in abstracts dataset")
                if abstract_record.get('abstract'):
                    return {
                        'text': abstract_record['abstract'],
                        'source': 'abstract',
                        'pdf_hash': None,
                        'status': 'ok',
                        'lookup_details': {
                            'corpus_id': str(corpus_id),
                            'release_id': self.current_release,
                            'has_local_data': bool(self.has_local_data),
                            'attempts': attempts + [self._attempt_result('abstracts', 'found_text')],
                        },
                    }
                else:
                    logger.info("Abstract record found but no abstract text available")
                    attempts.append(self._attempt_result('abstracts', 'record_without_text', 'abstract missing'))
            else:
                attempts.append(self._attempt_result('abstracts', 'missing_record'))


            # Try TLDR dataset
            logger.info("Attempting TLDR lookup...")
            tldr_record = self.indexer.lookup(
                release_id=self.current_release,
                dataset='tldrs',
                id_type='corpus_id', 
                search_id=str(corpus_id)
            )

            if tldr_record:
                logger.info("Found record in TLDR dataset")
                if tldr_record.get('text'):
                    return {
                        'text': tldr_record['text'],
                        'source': 'tldr',
                        'pdf_hash': None,
                        'status': 'ok',
                        'lookup_details': {
                            'corpus_id': str(corpus_id),
                            'release_id': self.current_release,
                            'has_local_data': bool(self.has_local_data),
                            'attempts': attempts + [self._attempt_result('tldrs', 'found_text')],
                        },
                    }
                else:
                    logger.info("TLDR record found but no text available")
                    attempts.append(self._attempt_result('tldrs', 'record_without_text', 'text missing'))
            else:
                attempts.append(self._attempt_result('tldrs', 'missing_record'))


            # If no content found in any dataset
            logger.warning(f"No content found for corpus ID: {corpus_id}")
            return self._build_inaccessible_result(
                corpus_id=str(corpus_id),
                reason_code='no_accessible_content',
                reason='No accessible text was found in S2ORC, abstracts, or TLDR datasets.',
                attempts=attempts,
            )

        except Exception as e:
            logger.error(f"Error getting paper content for {corpus_id}: {str(e)}", exc_info=True)
            attempts.append(self._attempt_result('lookup', 'exception', str(e)))
            return self._build_inaccessible_result(
                corpus_id=str(corpus_id),
                reason_code='lookup_exception',
                reason='An exception occurred while retrieving paper content.',
                attempts=attempts,
                error=str(e),
            )

class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.last_request = 0
        self.min_interval = 1.0 / requests_per_second

    async def wait(self):
        """Wait if necessary to maintain the rate limit."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            await asyncio.sleep(sleep_time)
        self.last_request = time.time()
