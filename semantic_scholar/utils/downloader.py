import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import math
from datetime import datetime
from tqdm import tqdm
import gzip
import shutil
import hashlib
from pathlib import Path
import sys
import time
from urllib.parse import urlparse, unquote
import re
import sqlite3
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)
from app.config.settings import Config

BASE_URL = "https://api.semanticscholar.org/datasets/v1"
console = Console()

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

class S2DatasetDownloader:
    def __init__(self):
        # Use project root for base directory
        self.base_dir = Path(project_root) / "semantic_scholar/datasets"
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(requests_per_second=0.5)  # Reduced to 1 request per 2 seconds
        
        self.api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY') or Config.SEMANTIC_SCHOLAR_API_KEY
        if not self.api_key:
            raise ValueError("No Semantic Scholar API key found. Set SEMANTIC_SCHOLAR_API_KEY in env_vars.json")
        
        self.session.headers.update({
            'x-api-key': self.api_key
        })
        
        self.datasets_to_download = [
            "papers", 
            "abstracts",
            "citations",
            "authors",
            "s2orc",
            "tldrs"
        ]
        
        # Create base and index directories with parents
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.base_dir / "indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Define which IDs to index for each dataset
        self.dataset_id_fields = {
            'papers': [
                ('paperId', 'paper_id'),
                ('corpusid', 'corpus_id')
            ],
            'abstracts': [('corpusid', 'corpus_id')],
            's2orc': [('corpusid', 'corpus_id')],
            'citations': [
                ('citingcorpusid', 'corpus_id'),
                ('citedcorpusid', 'corpus_id')
            ],
            'authors': [('authorid', 'author_id')],
            'tldrs': [('corpusid', 'corpus_id')]
        }

    def make_request(self, url: str, method: str = 'get', max_retries: int = 5, **kwargs) -> requests.Response:
        """Make a request with retry logic for rate limits and expired credentials."""
        for attempt in range(max_retries):
            try:
                # Wait for rate limit before making request
                self.rate_limiter.wait()
                
                # Handle headers properly - don't merge if pre-signed URL
                if 'AWSAccessKeyId' in url or 'x-amz-security-token' in url:
                    headers = kwargs.get('headers', {})
                else:
                    headers = {**self.session.headers, **(kwargs.get('headers', {}))}
                kwargs['headers'] = headers
                
                if method.lower() == 'get':
                    response = requests.get(url, **kwargs)
                elif method.lower() == 'head':
                    response = requests.head(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Handle different error cases
                if response.status_code == 429:  # Rate limit
                    wait_time = min(30, (2 ** attempt) + 1)
                    console.print(f"[yellow]Rate limited. Waiting {wait_time} seconds...[/yellow]")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    console.print(f"[yellow]Access denied. URL: {url}[/yellow]")
                    console.print(f"[yellow]Response: {response.text}[/yellow]")
                    raise requests.exceptions.HTTPError(f"403 Forbidden: {response.text}")
                
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = min(30, (2 ** attempt) + 1)
                console.print(f"[yellow]Request failed ({str(e)}). Retrying in {wait_time} seconds...[/yellow]")
                time.sleep(wait_time)

    def get_latest_release(self) -> str:
        """Get the latest release ID."""
        response = self.make_request(f"{BASE_URL}/release/latest")
        return response.json()["release_id"]

    def get_dataset_info(self, dataset_name: str, release_id: str) -> Dict:
        """Get information about a specific dataset including download links."""
        if dataset_name == 's2orc':
            # Special handling for S2ORC dataset
            url = f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/s2orc/"
            try:
                response = self.make_request(url)
                data = response.json()
                # Filter URLs to get only what we need for mini download
                if data.get('files'):
                    # Extract shard IDs for better progress tracking
                    data['files'] = [
                        {
                            'url': url,
                            'shard': re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url).group(2)
                        }
                        for url in data['files']
                    ]
                return data
            except requests.exceptions.HTTPError as e:
                console.print("[red]Error accessing S2ORC dataset. Make sure your API key has S2ORC access.[/red]")
                console.print("[yellow]For S2ORC access, visit: https://api.semanticscholar.org/s2orc[/yellow]")
                raise
        else:
            # Standard dataset handling
            url = f"{BASE_URL}/release/{release_id}/dataset/{dataset_name}"
            response = self.make_request(url)
            return response.json()

    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        if size_bytes == 0:
            return "Unknown size"
        
        size_names = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def estimate_dataset_size(self, url: str) -> int:
        """Estimate file size using HEAD request."""
        try:
            response = self.make_request(url, method='head')
            return int(response.headers.get('content-length', 0))
        except:
            return 0

    def get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL without query parameters."""
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        return os.path.basename(path)

    def _parallel_extract_gzip(self, input_path: Path, output_path: Path):
        """Revert to single-thread gzip extraction to avoid misalignment issues."""
        try:
            console.print(f"Extracting {input_path.name} without parallelization...")

            with gzip.open(input_path, 'rb') as gz_in, open(output_path, 'wb') as out:
                shutil.copyfileobj(gz_in, out)

        except Exception as e:
            console.print(f"[red]Error extracting {input_path.name}: {str(e)}[/red]")
            raise

    def verify_file(self, file_path: Path, expected_size: Optional[int] = None) -> bool:
        """Verify if a file is complete based on size."""
        if not file_path.exists():
            return False
        
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                console.print(f"[yellow]File {file_path.name} is incomplete (size: {actual_size} vs expected: {expected_size})[/yellow]")
                return False
        
        return True

    def download_file(self, url: str, output_dir: Path, desc: str = None) -> Tuple[bool, Optional[Path]]:
        """Download a file using wget with progress bar."""
        try:
            filename = self.get_filename_from_url(url)
            output_path = output_dir / filename
            desc = desc or f"Downloading {filename}"
            
            # If it's a gzipped file, we'll need both paths
            is_gzipped = filename.endswith('.gz')
            if is_gzipped:
                final_path = output_dir / filename.replace('.gz', '.json')
                if final_path.exists():
                    console.print(f"[green]File {final_path.name} already exists[/green]")
                    return True, final_path
            else:
                final_path = output_path
                if final_path.exists():
                    console.print(f"[green]File {final_path.name} already exists[/green]")
                    return True, final_path

            # Download using wget
            import subprocess
            try:
                console.print(f"[cyan]{desc}[/cyan]")
                subprocess.run(['wget', '-q', '--show-progress', url, '-O', str(output_path)], check=True)
                
                # Handle gzip extraction if needed
                if is_gzipped:
                    console.print(f"[cyan]Extracting {output_path.name}...[/cyan]")
                    with gzip.open(output_path, 'rb') as gz_in, open(final_path, 'wb') as out:
                        shutil.copyfileobj(gz_in, out)
                    # Remove the gzip file
                    output_path.unlink()
                    output_path = final_path
                
                console.print(f"[green]Successfully downloaded: {output_path}[/green]")
                return True, output_path
                
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to download file: {e}[/red]")
                if output_path.exists():
                    output_path.unlink()
                return False, None
                
        except Exception as e:
            console.print(f"[red]Error downloading file: {str(e)}[/red]")
            return False, None

    def _init_sqlite_db(self, index_path: Path):
        """Initialize SQLite database with proper schema and indices."""
        # First try to clean up any stale WAL files
        wal_file = index_path.parent / (index_path.name + "-wal")
        shm_file = index_path.parent / (index_path.name + "-shm")
        
        try:
            if wal_file.exists():
                wal_file.unlink()
            if shm_file.exists():
                shm_file.unlink()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove WAL files: {e}[/yellow]")

        max_retries = 3
        retry_delay = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(index_path), timeout=60) as conn:
                    # Enable WAL mode for better concurrent access
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size = -2000000")
                    conn.execute("PRAGMA page_size = 4096")
                    conn.execute("PRAGMA busy_timeout = 60000")
                    
                    # Create tables with proper indices
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS paper_locations (
                            id TEXT,              
                            id_type TEXT,         
                            dataset TEXT,         
                            file_path TEXT,       
                            line_offset INTEGER,  
                            PRIMARY KEY (id, id_type, dataset)
                        )
                    """)
                    
                    # Create indices for faster lookups
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_paper_locations_id 
                        ON paper_locations(id, id_type)
                    """)
                    
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_paper_locations_dataset 
                        ON paper_locations(dataset)
                    """)
                    
                    conn.commit()
                    return
                    
            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    console.print(f"[yellow]Database is locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(retry_delay)
                    continue
                raise
                
        if last_error:
            raise RuntimeError(f"Could not initialize database after {max_retries} attempts: {last_error}")

    def _get_db_connection(self, index_path: Path) -> sqlite3.Connection:
        """Get a database connection with retry logic."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(str(index_path), timeout=60)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size = -2000000")
                conn.execute("PRAGMA busy_timeout = 60000")  # 60 second busy timeout
                return conn
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    console.print(f"[yellow]Database is locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(retry_delay)
                    continue
                raise

    def _index_file(self, conn: sqlite3.Connection, file_path: Path, dataset: str):
        """Single-threaded file indexing with robust error handling."""
        try:
            # Remove any existing entries for this file
            conn.execute("DELETE FROM paper_locations WHERE file_path = ?", (str(file_path),))
            
            entries = []
            total_lines = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                offset = 0
                for line_num, line in enumerate(f, 1):
                    try:
                        if not line.strip():
                            continue
                            
                        item = json.loads(line.strip())
                        for field_name, id_type in self.dataset_id_fields[dataset]:
                            id_value = str(item.get(field_name, '')).lower()
                            if id_value:
                                entries.append((id_value, id_type, dataset, str(file_path), offset))
                                
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error processing line {line_num}: {str(e)}[/yellow]")
                        continue
                    
                    offset += len(line.encode('utf-8'))
                    total_lines += 1
                    
                    # Batch insert every 500k entries
                    if len(entries) >= 500000:
                        conn.execute('BEGIN IMMEDIATE')
                        try:
                            conn.executemany("""
                                INSERT OR REPLACE INTO paper_locations 
                                (id, id_type, dataset, file_path, line_offset)
                                VALUES (?, ?, ?, ?, ?)
                            """, entries)
                            conn.commit()
                            console.print(f"[green]Processed {total_lines:,} lines ({len(entries):,} entries)[/green]")
                            entries = []
                        except:
                            conn.rollback()
                            raise
                    
                    # Progress update every 500k lines
                    if line_num % 500000 == 0:
                        console.print(f"[cyan]Processed {line_num:,} lines...[/cyan]")
            
            # Insert any remaining entries
            if entries:
                conn.execute('BEGIN IMMEDIATE')
                try:
                    conn.executemany("""
                        INSERT OR REPLACE INTO paper_locations 
                        (id, id_type, dataset, file_path, line_offset)
                        VALUES (?, ?, ?, ?, ?)
                    """, entries)
                    conn.commit()
                except:
                    conn.rollback()
                    raise
            
            console.print(f"[bold green]✓ Successfully indexed {total_lines:,} total lines from {file_path.name}[/bold green]\n")
            
        except Exception as e:
            console.print(f"[red]Error indexing {file_path.name}: {str(e)}[/red]")
            raise

    def download_dataset(self, dataset_name: str, release_id: str = 'latest', mini: bool = False, index: bool = True) -> bool:
        """Download a specific dataset and optionally build index."""
        try:
            if release_id == 'latest':
                # First try to get local latest release
                local_release = self._get_latest_local_release()
                if local_release:
                    release_id = local_release
                    console.print(f"[cyan]Using latest local release: {release_id}[/cyan]")
                else:
                    # If no local release, get latest from API
                    release_id = self.get_latest_release()
                    console.print(f"[cyan]Using latest API release: {release_id}[/cyan]")
            
            # Get dataset info
            console.print(f"\n[cyan]Getting dataset info for {dataset_name}...[/cyan]")
            dataset_info = self.get_dataset_info(dataset_name, release_id)
            if not dataset_info:
                return False
            
            # Save metadata
            dataset_dir = self.base_dir / release_id / dataset_name
            os.makedirs(dataset_dir, exist_ok=True)
            
            metadata_path = dataset_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)

            # First check which files we need
            missing_files = []
            if dataset_name == 's2orc':
                files = dataset_info['files'][:1] if mini else dataset_info['files']
                console.print(f"\n[bold]Checking {len(files)} files for {dataset_name}...[/bold]")
                for file_info in files:
                    output_path = dataset_dir / f"{file_info['shard']}.json"
                    if output_path.exists():
                        size = output_path.stat().st_size / (1024 * 1024)
                        console.print(f"[green]✓ Already exists ({size:.1f} MB): {output_path.name}[/green]")
                    else:
                        console.print(f"[yellow]→ Needs download: {output_path.name}[/yellow]")
                        missing_files.append(file_info)
            else:
                files_to_download = dataset_info['files'][:1] if mini else dataset_info['files']
                console.print(f"\n[bold]Checking {len(files_to_download)} files for {dataset_name}...[/bold]")
                for file_url in files_to_download:
                    output_path = dataset_dir / self.get_filename_from_url(file_url).replace('.gz', '.json')
                    if output_path.exists():
                        size = output_path.stat().st_size / (1024 * 1024)
                        console.print(f"[green]✓ Already exists ({size:.1f} MB): {output_path.name}[/green]")
                    else:
                        console.print(f"[yellow]→ Needs download: {output_path.name}[/yellow]")
                        missing_files.append(file_url)

            # If we have missing files, get fresh dataset info and download them
            if missing_files:
                console.print(f"\n[cyan]Getting fresh download URLs for {len(missing_files)} missing files...[/cyan]")
                dataset_info = self.get_dataset_info(dataset_name, release_id)
                
                # Download missing files
                downloaded_files = []
                for file_info in missing_files:
                    if dataset_name == 's2orc':
                        url = file_info['url']
                        shard = file_info['shard']
                        success, path = self.download_file(url, dataset_dir, f"Downloading S2ORC shard {shard}")
                    else:
                        success, path = self.download_file(file_info, dataset_dir)
                    if success and path:
                        downloaded_files.append(path)

                if index and downloaded_files:
                    self.index_dataset(dataset_name, release_id, downloaded_files)
                
            return True
                
        except Exception as e:
            console.print(f"[red]Error downloading dataset {dataset_name}: {str(e)}[/red]")
            return False

    def index_dataset(self, dataset_name: str, release_id: str = 'latest', files: List[Path] = None, repair: bool = False) -> bool:
        """Index all files for a specific dataset."""
        try:
            # Get actual release ID if 'latest'
            if release_id == 'latest':
                release_id = self._get_latest_local_release()
                if not release_id:
                    console.print("[red]No local releases found. Please download datasets first.[/red]")
                    return False
                console.print(f"[cyan]Using latest local release: {release_id}[/cyan]")

            # Initialize SQLite index
            index_dir = self.base_dir / "indices"
            index_dir.mkdir(exist_ok=True)
            index_path = index_dir / f"{release_id}.db"
            
            # Clean up any existing WAL files before starting
            wal_file = index_path.parent / (index_path.name + "-wal")
            shm_file = index_path.parent / (index_path.name + "-shm")
            try:
                if wal_file.exists():
                    wal_file.unlink()
                if shm_file.exists():
                    shm_file.unlink()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not remove WAL files: {e}[/yellow]")
            
            # Initialize database schema if needed
            self._init_sqlite_db(index_path)
            
            # Get list of files to process
            if files is None:
                dataset_dir = self.base_dir / release_id / dataset_name
                if not dataset_dir.exists():
                    console.print(f"[yellow]Dataset directory not found: {dataset_dir}[/yellow]")
                    return False
                    
                files = [
                    f for f in dataset_dir.glob("*.json")
                    if f.name != 'metadata.json'
                ]
                
                if not files:
                    console.print(f"[yellow]No JSON files found in {dataset_dir}[/yellow]")
                    return False

            # Process each file with its own connection
            for file_path in files:
                try:
                    # Create a new connection for each file
                    with sqlite3.connect(str(index_path), timeout=300) as conn:  # 5 minute timeout
                        conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute("PRAGMA synchronous=NORMAL")
                        conn.execute("PRAGMA cache_size=-2000000")
                        conn.execute("PRAGMA busy_timeout=300000")  # Match the connection timeout
                        
                        # Check if file needs processing
                        if not repair:
                            cursor = conn.execute(
                                "SELECT COUNT(*) FROM paper_locations WHERE file_path = ?", 
                                (str(file_path),)
                            )
                            count = cursor.fetchone()[0]
                            if count > 0:
                                console.print(f"[green]File already indexed: {file_path.name} ({count:,} entries)[/green]")
                                continue
                        
                        # Process the file
                        console.print(f"[cyan]Indexing {file_path.name}...[/cyan]")
                        self._index_file(conn, file_path, dataset_name)
                        
                except Exception as e:
                    console.print(f"[red]Error processing {file_path.name}: {str(e)}[/red]")
                    if not repair:
                        raise
            
            return True
                
        except Exception as e:
            console.print(f"[red]Error indexing dataset {dataset_name}: {str(e)}[/red]")
            return False

    def download_all_datasets(self, release_id: str = 'latest', mini: bool = False):
        """Download all datasets first, then index them all."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"[bold cyan]Downloading all datasets for release {release_id}...[/bold cyan]")
        
        # First download all datasets without indexing
        for dataset in self.datasets_to_download:
            console.print(f"\n[bold]Downloading {dataset}...[/bold]")
            self.download_dataset(dataset, release_id, mini, index=False)
        
        # Then index all datasets
        console.print(f"\n[bold cyan]Indexing all datasets...[/bold cyan]")
        for dataset in self.datasets_to_download:
            console.print(f"\n[bold]Indexing {dataset}...[/bold]")
            self.index_dataset(dataset, release_id)

    def verify_downloads(self, mini: bool = True) -> bool:
        """Verify that all datasets were downloaded correctly."""
        try:
            release_id = self.get_latest_release()
            missing_files = {}
            
            for dataset in self.datasets_to_download:
                dataset_info = self.get_dataset_info(dataset, release_id)
                if not dataset_info:
                    continue
                    
                dataset_dir = self.base_dir / release_id / dataset
                if not dataset_dir.exists():
                    console.print(f"[red]Missing dataset directory: {dataset}[/red]")
                    continue
                
                # Get expected files
                if dataset == 's2orc':
                    expected_files = [
                        self.get_filename_from_url(file_info['url']).replace('.gz', '')
                        for file_info in (dataset_info['files'][:1] if mini else dataset_info['files'])
                    ]
                else:
                    expected_files = [
                        self.get_filename_from_url(url).replace('.gz', '')
                        for url in (dataset_info['files'][:1] if mini else dataset_info['files'])
                    ]
                
                # Get actual files
                actual_files = {
                    f.stem for f in dataset_dir.glob('*.json') 
                    if f.name != 'metadata.json'
                }
                
                # Find missing files
                missing = set(expected_files) - actual_files
                if missing:
                    missing_files[dataset] = missing
                    
            if missing_files:
                console.print("\nVerifying downloads for release {}...".format(release_id))
                for dataset, missing in missing_files.items():
                    console.print(f"Missing files in {dataset}: {missing}")
                return False
                
            return True
            
        except Exception as e:
            console.print(f"[red]Error verifying downloads: {str(e)}[/red]")
            raise

    def update_datasets(self) -> bool:
        """
        Update all datasets to the latest release using diffs.
        Automatically maintains indices during the update.
        """
        try:
            current_release = self._get_latest_local_release()
            if not current_release:
                console.print("[yellow]No local datasets found. Please run initial download first.[/yellow]")
                return False
            
            latest_release = self.get_latest_release()
            if current_release == latest_release:
                console.print("[green]Datasets already at latest release.[/green]")
                return True
            
            console.print(f"[cyan]Updating from {current_release} to {latest_release}...[/cyan]")
            
            # Process each dataset
            for dataset in self.datasets_to_download:
                try:
                    # Special handling for S2ORC dataset
                    if dataset == 's2orc':
                        try:
                            diff_url = f"https://api.semanticscholar.org/datasets/v1/diffs/{current_release}/to/{latest_release}/s2orc/"
                            response = self.make_request(diff_url)
                            diffs = response.json()
                            
                            # Filter and process S2ORC diffs
                            if 'diffs' in diffs:
                                for diff in diffs['diffs']:
                                    diff['update_files'] = [
                                        {
                                            'url': url,
                                            'shard': re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url).group(2)
                                        }
                                        for url in diff['update_files']
                                    ]
                                    diff['delete_files'] = [
                                        {
                                            'url': url,
                                            'shard': re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url).group(2)
                                        }
                                        for url in diff['delete_files']
                                    ]
                        except requests.exceptions.HTTPError as e:
                            console.print("[red]Error accessing S2ORC dataset. Make sure your API key has S2ORC access.[/red]")
                            console.print("[yellow]For S2ORC access, visit: https://api.semanticscholar.org/s2orc[/yellow]")
                            raise
                    else:
                        # Standard dataset handling
                        diff_url = f"{BASE_URL}/diffs/{current_release}/to/{latest_release}/{dataset}"
                        response = self.make_request(diff_url)
                        diffs = response.json()
                    
                    dataset_dir = self.base_dir / latest_release / dataset
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    
                    with Progress() as progress:
                        total_files = sum(
                            len(diff['update_files']) + len(diff['delete_files']) 
                            for diff in diffs['diffs']
                        )
                        task = progress.add_task(
                            f"[cyan]Updating {dataset}...", 
                            total=total_files
                        )
                        
                        # Process each diff sequentially
                        for diff in diffs['diffs']:
                            # Handle updates
                            update_files = diff['update_files']
                            if dataset == 's2orc':
                                update_files = [f['url'] for f in update_files]
                                
                            for url in update_files:
                                temp_file = dataset_dir / f"temp_{Path(url).name}"
                                try:
                                    success, output_path = self.download_file(url, dataset_dir)
                                    if success and output_path:
                                        # Update index with new/updated records
                                        self._update_index_for_file(output_path, dataset, latest_release)
                                finally:
                                    if temp_file.exists():
                                        temp_file.unlink()
                                progress.advance(task)
                            
                            # Handle deletes
                            delete_files = diff['delete_files']
                            if dataset == 's2orc':
                                delete_files = [f['url'] for f in delete_files]
                                
                            for url in delete_files:
                                temp_file = dataset_dir / f"temp_delete_{Path(url).name}"
                                try:
                                    success, output_path = self.download_file(url, dataset_dir)
                                    if success and output_path:
                                        # Remove deleted records from index
                                        self._remove_from_index(output_path, dataset)
                                        output_path.unlink()  # Remove the delete file after processing
                                finally:
                                    if temp_file.exists():
                                        temp_file.unlink()
                                progress.advance(task)
                        
                        # Verify index integrity after updates
                        console.print(f"[cyan]Verifying {dataset} index after update...[/cyan]")
                        self.verify_index_completeness()
                        
                    console.print(f"[green]Dataset {dataset} updated successfully![/green]")
                    
                except Exception as e:
                    console.print(f"[red]Error updating dataset {dataset}: {str(e)}[/red]")
                    return False
            
            console.print("[green]All datasets updated successfully![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error updating datasets: {str(e)}[/red]")
            return False

    def _update_index_for_file(self, file_path: Path, dataset: str, release_id: str):
        """Update index with new/updated records."""
        index_path = self.index_dir / f"{release_id}.db"
        
        with sqlite3.connect(str(index_path)) as conn:
            # Create tables if needed
            self._init_sqlite_db(index_path)
            
            # Process file line by line
            with open(file_path, 'r', encoding='utf-8') as f:
                offset = 0
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        corpus_id = str(record.get('corpusid')).lower()
                        
                        if corpus_id:
                            # Remove any existing entries for this corpus_id
                            conn.execute(
                                "DELETE FROM paper_locations WHERE id = ? AND dataset = ?",
                                (corpus_id, dataset)
                            )
                            
                            # Insert new entry
                            conn.execute(
                                """
                                INSERT INTO paper_locations 
                                (id, id_type, dataset, file_path, line_offset)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (corpus_id, 'corpus_id', dataset, str(file_path), offset)
                            )
                            
                    except json.JSONDecodeError:
                        console.print(f"[yellow]Warning: Invalid JSON in {file_path}[/yellow]")
                        
                    offset += len(line.encode('utf-8'))
                
                conn.commit()

    def _remove_from_index(self, file_path: Path, dataset: str):
        """Remove deleted records from index."""
        release_id = self._get_latest_local_release()
        index_path = self.index_dir / f"{release_id}.db"
        
        with sqlite3.connect(str(index_path)) as conn:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        corpus_id = str(record.get('corpusid')).lower()
                        
                        if corpus_id:
                            conn.execute(
                                "DELETE FROM paper_locations WHERE id = ? AND dataset = ?",
                                (corpus_id, dataset)
                            )
                            
                    except json.JSONDecodeError:
                        console.print(f"[yellow]Warning: Invalid JSON in {file_path}[/yellow]")
                
                conn.commit()

    def verify_index_completeness(self, datasets: List[str] = None) -> bool:
        """
        Verify that all downloaded files have been completely indexed.
        Uses parallel processing and optimized SQLite access.
        Automatically reindexes any corrupt or incomplete files.
        """
        try:
            release_id = self._get_latest_local_release()
            if not release_id:
                console.print("[yellow]No local datasets found.[/yellow]")
                return False

            # Use provided datasets or all datasets
            datasets = datasets or self.datasets_to_download

            index_path = self.index_dir / f"{release_id}.db"
            if not index_path.exists():
                console.print("[red]Index database not found.[/red]")
                return False

            def verify_file(args) -> Tuple[Path, str, int, int]:
                """Worker function to verify a single file's index completeness."""
                file_path, db_path, dataset = args
                if dataset not in datasets:  # Skip datasets not in our list
                    return None
                if file_path.name == 'metadata.json':
                    return None
                    
                # Count actual records in file
                actual_records = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        actual_records += 1
                        
                # Query index count for this file
                with sqlite3.connect(str(db_path)) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=-2000000")  # Use 2GB cache
                    
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM paper_locations 
                        WHERE file_path = ?
                    """, (str(file_path),))
                    indexed_records = cursor.fetchone()[0]
                    
                return (file_path, dataset, indexed_records, actual_records)

            incomplete_files = []
            num_cores = multiprocessing.cpu_count()
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Verifying index completeness...", total=0)
                
                for dataset in self.datasets_to_download if datasets is None else datasets:
                    dataset_dir = self.base_dir / release_id / dataset
                    if not dataset_dir.exists():
                        continue

                    json_files = list(dataset_dir.glob('*.json'))
                    progress.update(task, total=len(json_files))
                    
                    verify_args = [(f, index_path, dataset) for f in json_files]
                    
                    with ProcessPoolExecutor(max_workers=num_cores) as executor:
                        for result in executor.map(verify_file, verify_args):
                            if result is None:  # Skip metadata.json
                                continue
                                
                            file_path, dataset, indexed_count, actual_count = result
                            
                            if indexed_count == 0:
                                console.print(f"[red]File {file_path.name} has no index entries - will reindex[/red]")
                                incomplete_files.append((file_path, 'missing'))
                            elif indexed_count < actual_count:
                                console.print(
                                    f"[yellow]File {file_path.name} is partially indexed "
                                    f"({indexed_count}/{actual_count} records) - will reindex[/yellow]"
                                )
                                incomplete_files.append((file_path, 'partial'))
                                
                            progress.advance(task)

            if incomplete_files:
                console.print("\n[cyan]Reindexing incomplete files...[/cyan]")
                with sqlite3.connect(str(index_path)) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=-2000000")
                    
                    for file_path, _ in incomplete_files:
                        # Remove existing entries
                        conn.execute(
                            "DELETE FROM paper_locations WHERE file_path = ?", 
                            (str(file_path),)
                        )
                        # Reindex the file
                        dataset = file_path.parent.name
                        self._index_file(conn, file_path, dataset)
                console.print("[green]Reindexing complete![/green]")
                return True  # Return True since we fixed the issues
                
            console.print("[green]All files are properly indexed![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error verifying index: {str(e)}[/red]")
            return False

    def _get_latest_local_release(self) -> Optional[str]:
        """Get the latest release ID from local datasets directory."""
        if not self.base_dir.exists():
            return None
        
        # Get all subdirectories that look like release IDs (YYYY-MM-DD)
        releases = [
            d.name for d in self.base_dir.iterdir() 
            if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', d.name)
        ]
        
        if not releases:
            return None
        
        # Sort by date and return the latest
        return sorted(releases)[-1]

    def audit_datasets(self, release_id: str = 'latest', datasets: List[str] = None):
        """Audit dataset files and indexing status."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"\n[bold cyan]Auditing datasets for release {release_id}...[/bold cyan]")
        
        # Use provided datasets or all datasets
        datasets = datasets or self.datasets_to_download
        
        # Get index database
        index_path = self.base_dir / "indices" / f"{release_id}.db"
        if not index_path.exists():
            console.print("[red]No index database found for this release[/red]")
            return
        
        table = Table(
            "Dataset", 
            "Expected Files", 
            "Downloaded Files",
            "Indexed Files",
            "Status",
            title=f"Dataset Audit for Release {release_id}"
        )
        
        with sqlite3.connect(str(index_path)) as conn:
            for dataset in datasets:  # Changed from self.datasets_to_download to datasets
                try:
                    # Get expected files from API
                    dataset_info = self.get_dataset_info(dataset, release_id)
                    if not dataset_info:
                        table.add_row(
                            dataset,
                            "?",
                            "0",
                            "0",
                            "[red]Cannot fetch dataset info[/red]"
                        )
                        continue
                    
                    expected_count = len(dataset_info['files'])
                    
                    # Get downloaded files
                    dataset_dir = self.base_dir / release_id / dataset
                    if dataset == 's2orc':
                        downloaded_files = list(dataset_dir.glob("*.json")) if dataset_dir.exists() else []
                    else:
                        downloaded_files = [
                            f for f in dataset_dir.glob("*.json")
                            if f.name != 'metadata.json'
                        ] if dataset_dir.exists() else []
                    
                    # Get indexed files
                    cursor = conn.execute("""
                        SELECT DISTINCT file_path, COUNT(*) as entry_count 
                        FROM paper_locations 
                        WHERE dataset = ?
                        GROUP BY file_path
                    """, (dataset,))
                    indexed_files = {Path(path): count for path, count in cursor}
                    
                    # Determine status
                    if not dataset_dir.exists():
                        status = "[red]Not downloaded[/red]"
                    elif len(downloaded_files) < expected_count:
                        status = f"[yellow]Partially downloaded ({len(downloaded_files)}/{expected_count})[/yellow]"
                    elif len(indexed_files) < len(downloaded_files):
                        status = f"[yellow]Partially indexed ({len(indexed_files)}/{len(downloaded_files)})[/yellow]"
                    else:
                        status = "[green]Complete[/green]"
                    
                    # Add details for partially indexed files
                    partially_indexed = []
                    for f in downloaded_files:
                        if f in indexed_files and indexed_files[f] < 100:  # Arbitrary threshold
                            partially_indexed.append(f.name)
                    
                    if partially_indexed:
                        status += f"\n[yellow]Low index count: {', '.join(partially_indexed)}[/yellow]"
                    
                    table.add_row(
                        dataset,
                        str(expected_count),
                        str(len(downloaded_files)),
                        str(len(indexed_files)),
                        status
                    )
                    
                except Exception as e:
                    table.add_row(
                        dataset,
                        "?",
                        "?",
                        "?",
                        f"[red]Error: {str(e)}[/red]"
                    )
        
        console.print(table)

    def count_indices(self, release_id: str = 'latest'):
        """Print detailed index counts for each file."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"\n[bold cyan]Index counts for release {release_id}...[/bold cyan]")
        
        # Get index database
        index_path = self.base_dir / "indices" / f"{release_id}.db"
        if not index_path.exists():
            console.print("[red]No index database found for this release[/red]")
            return
        
        with sqlite3.connect(str(index_path)) as conn:
            # Get total count
            cursor = conn.execute("SELECT COUNT(*) FROM paper_locations")
            total_count = cursor.fetchone()[0]
            console.print(f"\nTotal index entries: [bold cyan]{total_count:,}[/bold cyan]\n")
            
            # Get counts by dataset
            table = Table(
                "Dataset",
                "File",
                "Index Count",
                "File Size",
                "Entries/MB",
                title="Index Counts by File"
            )
            
            cursor = conn.execute("""
                SELECT 
                    dataset,
                    file_path,
                    COUNT(*) as entry_count
                FROM paper_locations 
                GROUP BY dataset, file_path
                ORDER BY dataset, file_path
            """)
            
            current_dataset = None
            for dataset, file_path, count in cursor:
                # Add separator between datasets
                if current_dataset != dataset:
                    if current_dataset is not None:
                        table.add_row("", "", "", "", "")
                    current_dataset = dataset
                
                # Get file size if file exists
                path = Path(file_path)
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)  # Convert to MB
                    density = count / size_mb
                    size_str = f"{size_mb:.1f} MB"
                    density_str = f"{density:.1f}"
                else:
                    size_str = "[red]Missing[/red]"
                    density_str = "N/A"
                
                table.add_row(
                    dataset,
                    path.name,
                    f"{count:,}",
                    size_str,
                    density_str
                )
        
            console.print(table)

    def _verify_db_name(self):
        """Verify index database has correct name."""
        index_dir = self.base_dir / "indices"
        if not index_dir.exists():
            return
        
        db_files = list(index_dir.glob("*.db"))
        expected_name = f"{self._get_latest_local_release()}.db"
        
        for db_file in db_files:
            if db_file.name != expected_name:
                console.print(f"[yellow]Warning: Found database with incorrect name: {db_file.name}[/yellow]")
                console.print(f"[yellow]Expected name: {expected_name}[/yellow]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download Semantic Scholar datasets')
    parser.add_argument('--release', default='latest', help='Release ID to download')
    parser.add_argument('--mini', action='store_true', help='Download minimal dataset for testing')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded datasets')
    parser.add_argument('--verify-index', nargs='*', help='Verify index completeness. Optionally specify datasets to verify (e.g. --verify-index papers abstracts)')
    parser.add_argument('--audit', nargs='*', help='Audit datasets and indexing status. Optionally specify datasets to audit (e.g. --audit papers abstracts)')
    parser.add_argument('--index-only', nargs='*', help='Only run indexing on downloaded files. Optionally specify datasets to index (e.g. --index-only papers abstracts)')
    parser.add_argument('--download-only', action='store_true', help='Only download files without indexing')
    parser.add_argument('--repair', action='store_true', help='Repair/resume incomplete indexes')
    parser.add_argument('--count', action='store_true', help='Show detailed index counts for each file')
    args = parser.parse_args()
    
    downloader = S2DatasetDownloader()
    
    def validate_datasets(dataset_list):
        """Helper function to validate dataset names and return filtered list"""
        if not dataset_list:
            return downloader.datasets_to_download
            
        invalid_datasets = [d for d in dataset_list if d not in downloader.datasets_to_download]
        if invalid_datasets:
            console.print(f"[red]Invalid dataset names: {', '.join(invalid_datasets)}[/red]")
            console.print(f"[yellow]Valid datasets are: {', '.join(downloader.datasets_to_download)}[/yellow]")
            return None
        return dataset_list
    
    if args.count:
        downloader.count_indices(args.release)
    elif args.audit is not None:
        datasets = validate_datasets(args.audit)
        if datasets is None:
            return
        console.print(f"[cyan]Auditing datasets: {', '.join(datasets)}[/cyan]")
        downloader.audit_datasets(args.release, datasets)
    elif args.verify_index is not None:
        datasets = validate_datasets(args.verify_index)
        if datasets is None:
            return
        console.print(f"[cyan]Verifying index for datasets: {', '.join(datasets)}[/cyan]")
        downloader.verify_index_completeness(datasets)
    elif args.verify:
        downloader.verify_downloads(args.release)
    elif args.index_only is not None:
        datasets = validate_datasets(args.index_only)
        if datasets is None:
            return
        console.print(f"[cyan]Indexing datasets: {', '.join(datasets)}[/cyan]")
        for dataset in datasets:
            downloader.index_dataset(dataset, args.release, repair=args.repair)
    elif args.download_only:
        # Download all datasets without indexing
        for dataset in downloader.datasets_to_download:
            downloader.download_dataset(dataset, args.release, args.mini, index=False)
    else:
        downloader.download_all_datasets(args.release, args.mini)

if __name__ == "__main__":
    main() 