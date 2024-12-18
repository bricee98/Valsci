from typing import List, Union, Tuple
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
import signal
from functools import partial
from app.services.openai_service import OpenAIService
from textwrap import dedent
import threading
from fake_useragent import UserAgent
from http.cookiejar import LWPCookieJar
import gzip
import zlib
from io import BytesIO
import chardet
import brotli  # Make sure to install this: pip install brotli
from brotli import error as BrotliError
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiteratureSearcher:
    def __init__(self):
        self.email = Config.USER_EMAIL
        self.session = requests.Session()
        
        # Create a directory for cookies if it doesn't exist
        cookie_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cookies')
        os.makedirs(cookie_dir, exist_ok=True)
        
        # Set up the cookie jar with a file
        cookie_file = os.path.join(cookie_dir, 'literature_searcher_cookies.txt')
        self.session.cookies = LWPCookieJar(cookie_file)
        
        # Load existing cookies if the file exists
        if os.path.exists(cookie_file):
            self.session.cookies.load(ignore_discard=True, ignore_expires=True)
        
        self.user_agent = UserAgent()
        self.setup_session()
        self.openai_service = OpenAIService()  # Initialize OpenAIService
        self.pdf_session = requests.Session()
        self.setup_pdf_session()

        self.saved_search_queries = []

    def setup_session(self):
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def setup_pdf_session(self):
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.pdf_session.mount('https://', HTTPAdapter(max_retries=retries))
        self.pdf_session.mount('http://', HTTPAdapter(max_retries=retries))

    def make_api_request(self, url: str, params: dict = None, headers: dict = None, method: str = 'get', max_retries: int = 3) -> requests.Response:
        if params is None:
            params = {}
        params['email'] = self.email

        if headers is None:
            headers = {}

        # Set up headers to mimic a browser
        default_headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        headers = {**default_headers, **headers}

        # Construct the full URL with parameters
        full_url = url
        if params:
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query_string}"

        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request to {full_url} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Add a referer header sometimes
                if random.random() < 0.7:
                    headers['Referer'] = 'https://www.google.com/'

                if method.lower() == 'get':
                    response = self.session.get(url, params=params, headers=headers, allow_redirects=True, timeout=30)
                elif method.lower() == 'post':
                    response = self.session.post(url, json=params, headers=headers, allow_redirects=True, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()

                logger.info(f"Response headers: {response.headers}")
                logger.info(f"Response encoding: {response.encoding}")

                # Try to get the content without decoding
                content = response.content

                # Check if the content is compressed
                content_encoding = response.headers.get('Content-Encoding', '').lower()
                if content_encoding == 'gzip':
                    logger.info("Decompressing gzip content")
                    content = gzip.decompress(content)
                elif content_encoding == 'deflate':
                    logger.info("Decompressing deflate content")
                    content = zlib.decompress(content)
                elif content_encoding == 'br':
                    logger.info("Decompressing brotli content")
                    try:
                        content = brotli.decompress(content)
                    except BrotliError as e:
                        logger.error(f"Brotli decompression failed: {str(e)}")
                        # If Brotli decompression fails, try to use the raw content
                        content = response.content

                # Try to detect the encoding
                detected_encoding = chardet.detect(content)['encoding']
                logger.info(f"Detected encoding: {detected_encoding}")

                # Try to decode the content
                try:
                    decoded_content = content.decode(detected_encoding or 'utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode with {detected_encoding}, falling back to ISO-8859-1")
                    decoded_content = content.decode('iso-8859-1')

                logger.info(f"Decoded content (first 500 chars): {decoded_content[:500]}")

                # Create a new response object with the decoded content
                new_response = requests.Response()
                new_response.status_code = response.status_code
                new_response.headers = response.headers
                new_response._content = decoded_content.encode('utf-8')
                new_response.encoding = 'utf-8'

                # Check if the response is empty
                if not new_response.text.strip():
                    raise ValueError("Empty response received from the server")

                if response.status_code == 429:
                    wait_time = min(15, max(5, (2 ** attempt) + (random.randint(0, 1000) / 1000)))
                    logger.warning(f"Rate limit reached for {full_url}. Status code: {response.status_code}. Waiting for {wait_time:.2f} seconds before retrying.")
                    time.sleep(wait_time)
                    continue
                
                # Save cookies
                self.session.cookies.save(ignore_discard=True, ignore_expires=True)
                
                time.sleep(random.uniform(0.1, 0.4))  # Random delay between requests

                return new_response
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
            except ValueError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Empty response received from {full_url} after {max_retries} attempts.")
                    raise e
                wait_time = min(60, (2 ** attempt) + (random.randint(0, 1000) / 1000))
                logger.warning(f"Empty response from {full_url}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    def analyze_claim(self, claim: Claim) -> str:
        system_prompt = dedent("""
        You are an expert scientific analyst. Your task is to analyze the given claim and provide potential rationale or evidence that could support or refute it. Consider various aspects such as biological mechanisms, known associations, and potential research areas. Provide a detailed analysis without generating specific search queries.
        """).strip()

        user_message = f"Analyze the following claim and provide potential supporting or refuting evidence: {claim.text}"

        response = self.openai_service.generate_text(user_message, system_prompt=system_prompt)
        return response

    def generate_search_queries_from_analysis(self, claim: Claim, analysis: str, num_queries: int) -> List[str]:
        system_prompt = dedent("""
        You are SciSearchBot, an expert in generating academic search queries. Based on the given claim and its analysis, provide a list of {num_queries} search queries for OpenAlex to find relevant academic papers. These queries should help investigate the claims made in the analysis and explore related concepts. Return a JSON object with a single key 'queries' containing a list of search query strings.
        """).strip()

        user_message = f"Claim: {claim.text}\n\nAnalysis: {analysis}\n\nGenerate {num_queries} search queries to investigate this claim and analysis using OpenAlex."

        response = self.openai_service.generate_json(user_message, system_prompt=system_prompt)
        return response.get('queries', [])

    def generate_search_queries(self, claim: Claim, num_queries: int) -> dict:
        # Step 1: Analyze the claim
        analysis = self.analyze_claim(claim)

        # Step 2: Generate search queries based on the analysis
        search_queries = self.generate_search_queries_from_analysis(claim, analysis, num_queries)
        self.saved_search_queries.extend(search_queries)
        return {
            'explanation': analysis,
            'search_queries': search_queries
        }

    def search_papers(self, claim: Claim) -> List[Paper]:
        num_queries = claim.search_config.get('num_queries', 10)
        # Generate search queries using OpenAI
        search_data = self.generate_search_queries(claim, num_queries)
        search_queries = search_data.get('search_queries', [])
        logger.info(f"Generated search queries: {search_queries}")

        # Get configuration from claim object
        search_config = getattr(claim, 'search_config', {})
        results_per_query = search_config.get('results_per_query', 1)


        all_papers = []
        for query in search_queries:
            url = "https://api.openalex.org/works"
            params = {
                "search": query,
                "per-page": results_per_query,
                "select": "id,title,authorships,publication_year,primary_location,locations,open_access,abstract_inverted_index"
            }
            
            try:
                response = self.make_api_request(url, params=params)
                
                # Log the raw response content for debugging
                logger.debug(f"Raw API response: {response.text}")
                
                response_data = response.json()

                if 'results' not in response_data:
                    logger.error(f"Unexpected response structure from OpenAlex API: {response_data}")
                    continue

                for item in response_data['results']:
                    # Extract abstract from inverted index if available
                    abstract = None
                    if 'abstract_inverted_index' in item and item['abstract_inverted_index']:
                        try:
                            # Create a list of (word, position) tuples for all word positions
                            word_positions = []
                            for word, positions in item['abstract_inverted_index'].items():
                                for position in positions:
                                    word_positions.append((word, position))
                            
                            # Sort by position and join words
                            abstract = ' '.join(word for word, _ in sorted(word_positions, key=lambda x: x[1]))
                            logger.debug(f"Reconstructed abstract: {abstract[:200]}...")
                        except Exception as e:
                            logger.error(f"Error reconstructing abstract: {str(e)}")

                    # Extract journal name from primary_location if available
                    journal_name = None
                    if item.get('primary_location') and item['primary_location'].get('source'):
                        journal_name = item['primary_location']['source'].get('display_name')

                    # Extract authors from authorships
                    authors = []
                    for authorship in item.get('authorships', []):
                        if 'author' in authorship:
                            author_id = authorship['author'].get('id')
                            # Normalize the author_id here as well (to be consistent)
                            if author_id:
                                author_id = author_id.replace('https://openalex.org/', '')
                            authors.append({
                                'name': authorship['author'].get('display_name'),
                                'authorId': author_id
                            })

                    # Create Paper object with abstract
                    paper = Paper(
                        id=item.get('id'),
                        title=item['title'],
                        authors=authors,
                        year=item.get('publication_year'),
                        journal=journal_name,
                        url=self._get_best_url(item),
                        abstract=abstract
                    )

                    all_papers.append(paper)

            except requests.exceptions.JSONDecodeError as e:
                logger.error(f"JSONDecodeError for query '{query}': {str(e)}")
                logger.error(f"Response content: {response.text}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for query '{query}': {str(e)}")
                continue

        # Remove duplicates based on paper ID
        unique_papers = {paper.id: paper for paper in all_papers}.values()
        return list(unique_papers)

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

    def download_pdf_with_redirect(self, url: str) -> Union[str, None]:
        try:
            # Enhanced headers to mimic a real browser more closely
            headers = {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/pdf,application/x-pdf,application/octet-stream',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
                'Referer': 'https://scholar.google.com/'
            }

            # Add random common browser headers
            if random.random() < 0.5:
                headers['Sec-Ch-Ua'] = '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"'
                headers['Sec-Ch-Ua-Mobile'] = '?0'
                headers['Sec-Ch-Ua-Platform'] = '"Windows"'

            for attempt in range(3):
                try:
                    response = self.pdf_session.get(
                        url, 
                        headers=headers, 
                        allow_redirects=True, 
                        timeout=30,
                        stream=True
                    )
                    response.raise_for_status()

                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                        pdf_path = os.path.join('papers', f'{hash(url)}.pdf')
                        os.makedirs('papers', exist_ok=True)

                        with open(pdf_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                        logger.info(f"Successfully downloaded PDF from {url}")
                        return pdf_path
                    else:
                        logger.warning(f"URL {url} returned content-type: {content_type}")

                        if 'text/html' in content_type:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            # Look for PDF links
                            for a in soup.find_all('a', href=True):
                                href = a['href']
                                if href.lower().endswith('.pdf'):
                                    pdf_link = urljoin(url, href)
                                    logger.info(f"Trying PDF link: {pdf_link}")
                                    result = self.download_pdf_with_redirect(pdf_link)
                                    if result:
                                        return result
                                    break

                        return None

                except requests.RequestException as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                    if attempt < 2:
                        headers['User-Agent'] = self.user_agent.random
                        time.sleep(random.uniform(1, 3))
                        continue
                    raise

            return None

        except requests.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def extract_abstract_from_html(self, url: str) -> Union[str, None]:
        """Attempts to extract an abstract from an HTML page."""
        try:
            headers = {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for potential abstract text
            potential_abstracts = []
            for element in soup.find_all(['p', 'div']):
                text = element.get_text(strip=True)
                if len(text.split()) > 50:  # Consider elements with more than 50 words
                    potential_abstracts.append(text)

            # Evaluate each potential abstract
            for text in potential_abstracts:
                confidence = self.openai_service.evaluate_abstract_confidence(text)
                logger.info(f"Evaluated text with confidence: {confidence}")

                if confidence > 0.8:  # Threshold for considering it as an abstract
                    logger.info(f"Found abstract in HTML with confidence {confidence}")
                    return text
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting abstract from HTML at {url}: {str(e)}")
            return None

    def fetch_paper_content(self, paper: Paper, claim: Claim) -> Tuple[Union[str, None], str]:
        """
        Attempts to fetch paper content, returns tuple of (content, access_info)
        content can be None if paper is inaccessible
        access_info describes how the content was accessed or why it wasn't accessible
        """
        # Get abstracts_only setting from claim's search_config
        abstracts_only = claim.search_config.get('abstractsOnly', True)
        
        if abstracts_only:
            logger.info(f"Only using abstracts for paper: {paper.title}")
            # Try to get abstract in order of preference
            if paper.abstract:
                return paper.abstract, 'abstract_only'
            
            if paper.id:
                abstract = self.fetch_paper_abstract(paper.id)
                if abstract:
                    return abstract, 'openalex_abstract'
            
            if paper.url:
                abstract = self.extract_abstract_from_html(paper.url)
                if abstract:
                    return abstract, 'extracted_abstract'
            
            logger.warning(f"Paper {paper.title} is inaccessible")
            return None, 'inaccessible'

        # If not abstracts only, proceed with full text attempt
        if paper.url:
            logger.info(f"Attempting to download PDF from {paper.url}")
            try:
                pdf_path = self.download_pdf_with_redirect(paper.url)
                if pdf_path:
                    content = self.extract_text_from_pdf(pdf_path)
                    if content and not content.startswith("Error extracting text"):
                        return content, 'full_text'
                    else:
                        logger.warning(f"Failed to extract text from PDF for paper: {paper.title}")
                else:
                    logger.warning(f"Failed to download PDF for paper: {paper.title}")
            except Exception as e:
                logger.error(f"Error accessing paper {paper.title}: {str(e)}")
        
        # Fall back to abstract if full text fails
        if paper.abstract:
            return paper.abstract, 'abstract_only'
        
        if paper.id:
            abstract = self.fetch_paper_abstract(paper.id)
            if abstract:
                return abstract, 'openalex_abstract'
        
        if paper.url:
            abstract = self.extract_abstract_from_html(paper.url)
            if abstract:
                return abstract, 'extracted_abstract'
        
        logger.warning(f"Paper {paper.title} is inaccessible")
        return None, 'inaccessible'

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        def extract_with_pymupdf4llm():
            try:
                return pymupdf4llm.to_markdown(pdf_path)
            except Exception as e:
                logger.error(f"Error extracting text with pymupdf4llm from {pdf_path}: {str(e)}")
                return None

        def extract_with_timeout(timeout):
            result = [None]
            def target():
                result[0] = extract_with_pymupdf4llm()
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                logger.warning(f"Extraction with pymupdf4llm timed out for {pdf_path}")
                return None
            return result[0]

        try:
            # Try pymupdf4llm first with a 45-second timeout
            full_text = extract_with_timeout(45)

            if full_text:
                logger.info(f"Successfully extracted text from {pdf_path} using pymupdf4llm")
            else:
                raise Exception("pymupdf4llm failed to extract text")

        except Exception as e:
            logger.warning(f"Falling back to PyPDF2 for {pdf_path}: {str(e)}")

            # Fall back to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    for page in reader.pages:
                        full_text += page.extract_text() + "\n\n"
                
                if not full_text.strip():
                    raise ValueError("No text extracted from PDF")
                
                logger.info(f"Successfully extracted text from {pdf_path} using PyPDF2")
            except Exception as e:
                logger.error(f"Error extracting text with PyPDF2 from {pdf_path}: {str(e)}")
                return f"Error extracting text from {pdf_path}: {str(e)}"

        # Save the extracted text to a file
        text_file_path = os.path.splitext(pdf_path)[0] + "_extracted.txt"
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(full_text)
        
        logger.info(f"Extracted text saved to: {text_file_path}")
        
        return full_text

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

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

