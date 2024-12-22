import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time
from rich.console import Console
import ijson
from openai import OpenAI
import asyncio
import mmap
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

project_root = str(Path(__file__).parent.parent.parent)
from app.config.settings import Config
from app.services.openai_service import OpenAIService

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
        self.openai_service = OpenAIService()
        self.saved_search_queries = []
        
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

    def generate_search_queries(self, claim_text: str, num_queries: int = 5) -> List[str]:
        """Generate search queries for a claim using GPT."""
        system_prompt = """
        You are an expert at converting scientific claims into effective search queries.
        Generate search queries that will find papers relevant to validating or refuting the claim.
        
        Guidelines:
        1. Focus on key scientific concepts and their relationships
        2. Include specific technical terms and their synonyms
        3. Break down complex claims into simpler search components
        4. Use boolean operators (AND, OR) when helpful
        5. Keep queries concise but precise
        6. Include both broad and specific variations
        7. Consider alternative terminology used in the field
        
        Format each query to maximize relevance for academic paper search.
        """

        user_prompt = f"""
        Generate {num_queries} different search queries for the following scientific claim:
        "{claim_text}"

        Make the queries specific enough to find relevant papers but not so narrow that they miss important results.
        Return the queries as a JSON array of strings.
        """

        response = self.openai_service.generate_json(user_prompt, system_prompt)
        queries = response.get('queries', [])
        
        # Log generated queries for debugging
        console.print("\n[cyan]Generated queries:[/cyan]")
        for query in queries:
            console.print(f"[green]- {query}[/green]")
            
        self.saved_search_queries.extend(queries)
        return queries

    def search_papers_for_claim(self, claim_text: str, 
                              num_queries: int = 5, 
                              results_per_query: int = 5) -> List[Dict]:
        """Search papers relevant to a claim."""
        papers = []
        seen_paper_ids = set()
        
        queries = self.generate_search_queries(claim_text, num_queries)
        
        for query in queries:
            try:
                search_results = self.search_papers(query, limit=results_per_query)
                
                for paper in search_results:
                    corpus_id = paper.get('corpusId')

                    console.print(f"[green]Corpus ID: {corpus_id}[/green]")
                    if corpus_id and corpus_id not in seen_paper_ids:
                        # Get full content
                        content = self.get_paper_content(corpus_id)
                        console.print(f"[green]Content: {content}[/green]")
                        if content:
                            paper['text'] = content['text']
                            paper['content_source'] = content['source']
                            paper['pdf_hash'] = content['pdf_hash']
                        
                        seen_paper_ids.add(corpus_id)
                        papers.append(paper)
                        
            except Exception as e:
                console.print(f"[red]Error in search_papers_for_claim: {str(e)}[/red]")
                continue

        print("are any papers none? ", any(paper is None for paper in papers))

        # remove None papers
        papers = [paper for paper in papers if paper is not None]

        return papers

    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search papers using S2 API and cross-reference with local data."""
        try:
            # Rate limiting
            time.sleep(1)  # Add 1 second delay before API call

            console.print(f"[green]Searching for papers with query: {query}[/green]")
            
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
            console.print(f"[red]Error in search_papers: {str(e)}[/red]")
            return []

    def _api_search(self, query: str, limit: int) -> List[Dict]:
        """Search papers using Semantic Scholar API."""
        self.rate_limiter.wait()
        response = self.session.get(
            'https://api.semanticscholar.org/graph/v1/paper/search',
            params={
                'query': query,
                'limit': limit,
                'fields': ','.join([
                    'paperId',
                    'title',
                    'abstract',
                    'year',
                    'authors',
                    'venue',
                    'url',
                    'isOpenAccess',
                    'fieldsOfStudy',
                    'citationCount',  # Add citation count from API
                    'authors.name',
                    'authors.authorId',
                    'authors.affiliations',
                    'authors.paperCount',
                    'authors.citationCount',
                    'authors.hIndex'
                ])
            }
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get('data'):
            console.print(f"[yellow]No results found for query: {query}[/yellow]")
            return []
            
        return data.get('data', [])

    def _get_citation_count(self, paper_id: str) -> int:
        """Get citation count for a paper."""
        citations_data = self._find_in_dataset('citations', paper_id)
        return len(citations_data.get('citations', [])) if citations_data else 0

    def _enrich_author_data(self, authors: List[Dict]) -> List[Dict]:
        """Add additional author information from local dataset."""
        enriched_authors = []
        for author in authors:
            author_id = author.get('authorId')
            if author_id:
                local_data = self._find_in_dataset('authors', author_id)
                if local_data:
                    # Add h-index and other metrics
                    author['hIndex'] = local_data.get('hIndex', 0)
                    author['paperCount'] = local_data.get('paperCount', 0)
                    author['citationCount'] = local_data.get('citationCount', 0)
            enriched_authors.append(author)
        return enriched_authors

    def _get_local_paper_data(self, paper_id: str) -> Optional[Dict]:
        """Get paper data from local dataset."""
        paper_data = self._find_in_dataset('papers', paper_id)
        if not paper_data:
            return None

        abstract_data = self._find_in_dataset('abstracts', paper_id)
        if abstract_data:
            paper_data['abstract'] = abstract_data.get('abstract')

        citations = self._find_in_dataset('citations', paper_id)
        if citations:
            paper_data['citations'] = citations.get('citations', [])

        return paper_data

    def _find_in_dataset(self, dataset: str, item_id: str, id_type: str = None) -> Optional[Dict]:
        """Look up an item in a local dataset using our BinaryIndexer."""
        logger.info(f"Looking up item in dataset: {dataset}, item_id: {item_id}, id_type: {id_type}")
        
        if not self.has_local_data:
            logger.warning("No local data available, skipping dataset lookup")
            return None
        
        try:
            # Map all known ID types to their JSON field names
            id_mappings = {
                'corpus_id': 'corpusid',  # Used by papers, abstracts, s2orc, tldrs
                'author_id': 'authorid'
            }
            logger.info(f"ID mappings: {id_mappings}")

            # For all paper-related datasets, default to corpus_id
            if dataset in ['papers', 'abstracts', 'tldrs', 's2orc'] and not id_type:
                id_type = 'corpus_id'
                logger.info(f"Using default id_type 'corpus_id' for dataset {dataset}")

            mapped_id_type = id_mappings.get(id_type, id_type)
            logger.info(f"Mapped id_type '{id_type}' to '{mapped_id_type}'")

            record = self.indexer.lookup(
                release_id=self.current_release,
                dataset=dataset,
                id_type=mapped_id_type,
                search_id=str(item_id).lower()
            )
            
            if record:
                logger.info(f"Found record in {dataset}")
            else:
                logger.info(f"No record found in {dataset}")
                
            return record
            
        except Exception as e:
            logger.error(f"Binary index lookup error: {str(e)}", exc_info=True)
            return None

    def get_paper_content(self, corpus_id: str) -> Optional[Dict]:
        """Get full paper content from S2ORC or abstract data, using the BinaryIndexer."""
        logger.info(f"Attempting to get content for corpus ID: {corpus_id}")
        logger.info(f"Current release: {self.current_release}")
        logger.info(f"Has local data: {self.has_local_data}")
        
        if not self.current_release:
            logger.warning("No release ID available")
            return None
        
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
                        'pdf_hash': s2orc_record.get('pdf_parse', {}).get('pdf_hash')
                    }
                else:
                    logger.info("S2ORC record found but no body text available")


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
                        'pdf_hash': None
                    }
                else:
                    logger.info("Abstract record found but no abstract text available")


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
                        'pdf_hash': None
                    }
                else:
                    logger.info("TLDR record found but no text available")


            # If no content found in any dataset
            logger.warning(f"No content found for corpus ID: {corpus_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting paper content for {corpus_id}: {str(e)}", exc_info=True)
            return None

class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.last_request = 0
        self.min_interval = 1.0 / requests_per_second

    def wait(self):
        """Wait if necessary to maintain the rate limit."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request = time.time()