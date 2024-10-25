from typing import List, Union
import os
import requests
from bs4 import BeautifulSoup
from app.models.claim import Claim
from app.models.paper import Paper, PaperMetadata
from app.config.settings import Config
import pymupdf4llm
import tempfile
import time
from requests.exceptions import RequestException
import random
import logging
import urllib.parse
import PyPDF2
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiteratureSearcher:
    def __init__(self):
        self.email = Config.USER_EMAIL
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def make_api_request(self, url: str, params: dict = None, headers: dict = None, method: str = 'get', max_retries: int = 7) -> requests.Response:
        if params is None:
            params = {}
        params['email'] = self.email

        if headers is None:
            headers = {}
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

        # Construct the full URL with parameters
        full_url = url
        if params:
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query_string}"

        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request to {full_url} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                if method.lower() == 'get':
                    response = self.session.get(url, params=params, headers=headers, allow_redirects=True)
                elif method.lower() == 'post':
                    response = self.session.post(url, json=params, headers=headers, allow_redirects=True)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()

                if response.status_code == 429:
                    wait_time = min(15, max(5, (2 ** attempt) + (random.randint(0, 1000) / 1000)))
                    logger.warning(f"Rate limit reached for {full_url}. Status code: {response.status_code}. Waiting for {wait_time:.2f} seconds before retrying.")
                    time.sleep(wait_time)
                    continue
                
                time.sleep(3)  # Wait 3 seconds between requests
                return response
            except RequestException as e:
                if attempt == max_retries - 1:
                    status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'Unknown'
                    error_message = str(e)
                    
                    # Check for 'error' key in the response
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_data = e.response.json()
                            if 'error' in error_data:
                                error_message = f"{error_message}. API Error: {error_data['error']}"
                        except ValueError:
                            pass  # Response body is not JSON
                    
                    logger.error(f"Max retries reached for {full_url}. Status code: {status_code}. Error: {error_message}")
                    
                    # If the URL starts with 'http://', try 'https://'
                    if url.startswith('http://'):
                        https_url = 'https://' + url[7:]
                        logger.info(f"Attempting HTTPS connection: {https_url}")
                        return self.make_api_request(https_url, params, headers, method, max_retries)
                    
                    raise e
                wait_time = min(60, (2 ** attempt) + (random.randint(0, 1000) / 1000))
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'Unknown'
                logger.warning(f"Request to {full_url} failed. Status code: {status_code}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    def search_papers(self, claim: Claim) -> List[Paper]:
        url = "https://api.openalex.org/works"
        params = {
            "search": claim.text,
            "per-page": 3,
            "select": "id,title,authorships,publication_year,primary_location,locations,open_access"
        }
        
        response = self.make_api_request(url, params=params)
        response_data = response.json()

        if 'results' not in response_data:
            logger.error(f"Unexpected response structure from OpenAlex API: {response_data}")
            return []

        papers = []
        for item in response_data['results']:
            # Extract journal name from primary_location if available
            journal_name = None
            if item.get('primary_location') and item['primary_location'].get('source'):
                journal_name = item['primary_location']['source'].get('display_name')

            # Extract authors from authorships
            authors = []
            for authorship in item.get('authorships', []):
                if 'author' in authorship:
                    authors.append({
                        'name': authorship['author'].get('display_name'),
                        'authorId': authorship['author'].get('id')
                    })

            # Create Paper object without abstract
            paper = Paper(
                id=item.get('id'),
                title=item['title'],
                authors=authors,
                year=item.get('publication_year'),
                journal=journal_name,
                url=self._get_best_url(item),
                abstract=None  # We'll fetch this separately if needed
            )

            # Fetch abstract if required
            if Config.FETCH_ABSTRACTS:
                try:
                    paper.abstract = self.fetch_paper_abstract(paper.id)
                    if paper.abstract is None:
                        logger.warning(f"Unable to fetch abstract for paper {paper.id}")
                except Exception as e:
                    logger.error(f"Error while fetching abstract for paper {paper.id}: {str(e)}")
                    paper.abstract = None

            papers.append(paper)
        
        return papers

    def _get_best_url(self, item: dict) -> str:
        """Get the best available URL for the paper."""
        # First check if there's an OA URL
        if item.get('open_access') and item['open_access'].get('oa_url'):
            return item['open_access']['oa_url']
        
        # Then check primary location
        if item.get('primary_location') and item['primary_location'].get('landing_page_url'):
            return item['primary_location']['landing_page_url']
        
        # Finally check other locations
        for location in item.get('locations', []):
            if location.get('landing_page_url'):
                return location['landing_page_url']
        
        return None

    def fetch_paper_metadata(self, paper_id: str) -> PaperMetadata:
        url = f"https://api.openalex.org/works/{paper_id}"
        params = {
            "select": "title,authorships,publication_year,primary_location,cited_by_count"
        }
        
        response = self.make_api_request(url, params=params)
        data = response.json()
        
        # Extract journal name from primary_location
        journal_name = None
        if data.get('primary_location') and data['primary_location'].get('source'):
            journal_name = data['primary_location']['source'].get('display_name')

        # Extract authors from authorships
        authors = []
        for authorship in data.get('authorships', []):
            if 'author' in authorship and 'display_name' in authorship['author']:
                authors.append(authorship['author']['display_name'])

        return PaperMetadata(
            title=data.get('title'),
            authors=authors,
            year=data.get('publication_year'),
            journal=journal_name,
            citation_count=data.get('cited_by_count'),
            influential_citation_count=None  # OpenAlex doesn't have this concept
        )

    def get_author_metrics(self, author_id: str) -> dict:
        """Get author metrics from OpenAlex."""
        url = f"https://api.openalex.org/authors/{author_id}"
        params = {
            "select": "summary_stats,works_count,cited_by_count"
        }
        
        response = self.make_api_request(url, params=params)
        data = response.json()
        
        return {
            'h_index': data.get('summary_stats', {}).get('h_index', 0),
            'works_count': data.get('works_count', 0),
            'cited_by_count': data.get('cited_by_count', 0)
        }

    # Keeping other methods unchanged
    def fetch_paper_content(self, paper: Paper) -> str:
        # First, try to get the PDF using the paper's URL
        if paper.url:
            pdf_path = self.download_pdf(paper.url)
            if pdf_path:
                content = self.extract_text_from_pdf(pdf_path)
                if content and not content.startswith("Error extracting text"):
                    return content
    
        
        # If all else fails, fall back to the abstract
        return paper.abstract

    def download_pdf(self, url: str) -> Union[str, None]:
        try:
            response = self.make_api_request(url, method='get')
            
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                pdf_path = os.path.join('papers', f'{hash(url)}.pdf')
                
                # Create the papers directory if it doesn't exist
                os.makedirs('papers', exist_ok=True)
                
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return pdf_path
            else:
                return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n\n"
            
            if not full_text.strip():
                raise ValueError("No text extracted from PDF")
            
            return full_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return f"Error extracting text from {pdf_path}: {str(e)}"

    def fetch_paper_abstract(self, paper_id: str) -> Union[str, None]:
        """Fetch the abstract for a given paper ID."""
        url = f"https://api.openalex.org/works/{paper_id}"
        params = {
            "select": "abstract_inverted_index"
        }
        
        try:
            response = self.make_api_request(url, params=params)
            data = response.json()
            
            if 'abstract_inverted_index' in data and data['abstract_inverted_index'] is not None:
                # Reconstruct the abstract from the inverted index
                abstract_words = sorted(data['abstract_inverted_index'].items(), key=lambda x: x[1][0])
                abstract = ' '.join(word for word, _ in abstract_words)
                return abstract
            else:
                logger.warning(f"No abstract found for paper {paper_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching abstract for paper {paper_id}: {str(e)}")
            return None
