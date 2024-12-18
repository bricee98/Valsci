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
        self.rate_limiter = RateLimiter(requests_per_second=1.0)
        
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
            'authors': [('authorid', 'author_id')]
        }

    def make_request(self, url: str, method: str = 'get', max_retries: int = 5, **kwargs) -> requests.Response:
        """Make a request with retry logic for rate limits and expired credentials."""
        for attempt in range(max_retries):
            try:
                # Wait for rate limit before making request
                self.rate_limiter.wait()
                
                if method.lower() == 'get':
                    response = self.session.get(url, **kwargs)
                elif method.lower() == 'head':
                    response = self.session.head(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Handle different error cases
                if response.status_code == 429:  # Rate limit
                    wait_time = min(30, (2 ** attempt) + 1)
                    console.print(f"[yellow]Rate limited. Waiting {wait_time} seconds...[/yellow]")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:  # Expired credentials
                    console.print("[yellow]URL expired. Refreshing dataset info...[/yellow]")
                    # Re-fetch the dataset info to get fresh pre-signed URLs
                    if hasattr(self, '_current_dataset'):
                        dataset_info = self.get_dataset_info(self._current_dataset, self._current_release)
                        if 'files' in dataset_info:
                            # Find matching new URL
                            old_filename = self.get_filename_from_url(url)
                            for new_url in dataset_info['files']:
                                if isinstance(new_url, dict):  # Handle S2ORC case
                                    new_url = new_url['url']
                                if self.get_filename_from_url(new_url) == old_filename:
                                    console.print("[green]Got fresh URL, retrying...[/green]")
                                    return self.make_request(new_url, method, max_retries-attempt, **kwargs)
                    raise
                
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

    def download_file(self, url: str, output_dir: Path, desc: str = None) -> Tuple[bool, Optional[Path]]:
        """Download a file with progress bar. Returns (success, output_path)."""
        try:
            filename = self.get_filename_from_url(url)
            output_path = output_dir / filename
            desc = desc or f"Downloading {filename}"
            
            response = self.make_request(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                
            # If file is gzipped, extract it using parallel processing
            if filename.endswith('.gz'):
                console.print(f"Extracting {filename}...")
                base_name = filename.replace('.gz', '')
                if not base_name.endswith('.json'):
                    base_name += '.json'
                
                output_json_path = output_dir / base_name
                self._parallel_extract_gzip(output_path, output_json_path)
                os.remove(output_path)  # Remove the gzipped file
                return True, output_json_path
            
            return True, output_path
            
        except Exception as e:
            console.print(f"[red]Error downloading {url}: {str(e)}[/red]")
            return False, None

    def _init_sqlite_db(self, index_path: Path):
        """Initialize SQLite database with proper schema and indices."""
        with sqlite3.connect(str(index_path)) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size = -2000000")
            conn.execute("PRAGMA page_size = 4096")
            
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

    def _parallel_index_chunk(self, chunk_data: Tuple[Path, str, int, int]) -> List[Tuple]:
        """Process a chunk of a file and return index entries."""
        file_path, dataset, start_pos, chunk_size = chunk_data
        entries = []
        
        console.print(f"[cyan]Processing chunk at offset {start_pos} of {file_path.name}...[/cyan]")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(start_pos)
            data = f.read(chunk_size)
            lines = data.splitlines()
            
            offset = start_pos
            for line_num, line in enumerate(lines, 1):
                try:
                    item = json.loads(line.strip())
                    for field_name, id_type in self.dataset_id_fields[dataset]:
                        id_value = str(item.get(field_name, '')).lower()
                        if id_value:
                            entries.append((id_value, id_type, dataset, str(file_path), offset))
                except json.JSONDecodeError:
                    continue
                offset += len(line.encode('utf-8')) + 1  # +1 for newline
                
                if line_num % 100000 == 0:
                    console.print(f"[cyan]Processed {line_num:,} lines in chunk...[/cyan]")
                
        console.print(f"[green]Completed chunk with {len(entries):,} entries[/green]")
        return entries

    def _index_file(self, conn: sqlite3.Connection, file_path: Path, dataset: str):
        """Parallel file indexing using all available cores."""
        num_cores = multiprocessing.cpu_count()
        file_size = file_path.stat().st_size
        chunk_size = file_size // num_cores
        
        console.print(f"\n[bold cyan]Starting parallel indexing of {file_path.name} using {num_cores} cores...[/bold cyan]")
        console.print(f"File size: {self.format_size(file_size)}, Chunk size: {self.format_size(chunk_size)}")
        
        chunks = [
            (file_path, dataset, i * chunk_size, chunk_size)
            for i in range(num_cores)
        ]
        
        # Add remainder to last chunk
        chunks[-1] = (file_path, dataset, (num_cores-1) * chunk_size, file_size - (num_cores-1) * chunk_size)
        
        try:
            conn.execute('BEGIN TRANSACTION')
            total_entries = 0
            
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                for i, chunk_entries in enumerate(executor.map(self._parallel_index_chunk, chunks), 1):
                    if chunk_entries:
                        conn.executemany("""
                            INSERT OR REPLACE INTO paper_locations 
                            (id, id_type, dataset, file_path, line_offset)
                            VALUES (?, ?, ?, ?, ?)
                        """, chunk_entries)
                        total_entries += len(chunk_entries)
                    console.print(f"[green]Processed chunk {i}/{num_cores} ({len(chunk_entries):,} entries)[/green]")
            
            conn.execute('COMMIT')
            console.print(f"[bold green]✓ Successfully indexed {total_entries:,} total entries from {file_path.name}[/bold green]\n")
            
        except Exception as e:
            conn.execute('ROLLBACK')
            console.print(f"[red]Error indexing {file_path.name}: {str(e)}[/red]")
            raise

    def download_dataset(self, dataset_name: str, release_id: str = 'latest', mini: bool = False) -> bool:
        """Download a specific dataset and build index."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        try:
            # Store current dataset context for credential refresh
            self._current_dataset = dataset_name
            self._current_release = release_id
            
            dataset_info = self.get_dataset_info(dataset_name, release_id)
            if not dataset_info:
                return False
            
            dataset_dir = self.base_dir / release_id / dataset_name
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save metadata
            metadata_path = dataset_dir / 'metadata.json'
            if not metadata_path.exists():
                with open(metadata_path, 'w') as f:
                    json.dump(dataset_info, f, indent=2)

            # Initialize SQLite index
            index_dir = self.base_dir / "indices"
            index_dir.mkdir(exist_ok=True)
            index_path = index_dir / f"{release_id}.db"
            
            # Initialize database schema if needed
            self._init_sqlite_db(index_path)
            
            with sqlite3.connect(str(index_path)) as conn:
                # Get list of already processed files
                processed_files = set()
                cursor = conn.execute(
                    "SELECT DISTINCT file_path FROM paper_locations WHERE dataset = ?", 
                    (dataset_name,)
                )
                for (file_path,) in cursor:
                    processed_files.add(Path(file_path).name)

                # Handle S2ORC differently
                if dataset_name == 's2orc':
                    files = dataset_info['files']
                    if mini:
                        files = files[:1]  # Get only first shard for mini download
                    
                    # Filter out already processed files
                    remaining_files = [
                        f for f in files 
                        if not (dataset_dir / f"{f['shard']}.json").exists() and
                        f"{f['shard']}.json" not in processed_files
                    ]
                    
                    if not remaining_files:
                        console.print(f"[green]All files already downloaded for {dataset_name}[/green]")
                        return True
                    
                    with Progress() as progress:
                        task = progress.add_task(
                            f"Downloading {dataset_name}...", 
                            total=len(files),
                            completed=len(files) - len(remaining_files)
                        )
                        
                        for file_info in remaining_files:
                            url = file_info['url']
                            shard = file_info['shard']
                            output_path = dataset_dir / f"{shard}.json"
                            
                            if self.download_file(url, dataset_dir, f"Downloading S2ORC shard {shard}"):
                                self._index_file(conn, output_path, dataset_name)
                            progress.advance(task)
                else:
                    # Standard dataset handling
                    files_to_download = dataset_info['files'][:1] if mini else dataset_info['files']
                    
                    # Filter out already processed files
                    remaining_files = [
                        url for url in files_to_download
                        if not (dataset_dir / self.get_filename_from_url(url).replace('.gz', '.json')).exists() and
                        self.get_filename_from_url(url).replace('.gz', '.json') not in processed_files
                    ]
                    
                    if not remaining_files:
                        console.print(f"[green]All files already downloaded for {dataset_name}[/green]")
                        return True
                    
                    with Progress() as progress:
                        task = progress.add_task(
                            f"Downloading {dataset_name}...", 
                            total=len(files_to_download),
                            completed=len(files_to_download) - len(remaining_files)
                        )
                        
                        for file_url in remaining_files:
                            success, output_path = self.download_file(file_url, dataset_dir)
                            if success and output_path:
                                console.print(f"[cyan]Starting indexing for {output_path.name}...[/cyan]")
                                self._index_file(conn, output_path, dataset_name)
                            progress.advance(task)
                    
                    conn.commit()
                return True
                
        except Exception as e:
            console.print(f"[red]Error downloading dataset {dataset_name}: {str(e)}[/red]")
            return False

    def download_all_datasets(self, release_id: str = 'latest', mini: bool = False) -> bool:
        """Download all specified datasets."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        console.print(f"[cyan]Downloading datasets for release {release_id}[/cyan]")
        console.print(f"[cyan]Files will be saved to: {self.base_dir}[/cyan]")
        
        success = True
        for dataset in self.datasets_to_download:
            console.print(f"\n[cyan]Downloading {dataset} dataset...[/cyan]")
            if not self.download_dataset(dataset, release_id, mini):
                success = False
                console.print(f"[red]Failed to download {dataset} dataset[/red]")
        
        if success:
            console.print("\n[green]All datasets downloaded successfully![/green]")
        else:
            console.print("\n[red]Some datasets failed to download[/red]")
        
        return success

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
        Returns True if update was successful.
        """
        try:
            # Get current and latest release IDs
            current_release = self._get_latest_local_release()
            if not current_release:
                console.print("[yellow]No local datasets found. Please run initial download first.[/yellow]")
                return False
            
            latest_release = self.get_latest_release()
            if current_release == latest_release:
                console.print("[green]Datasets already at latest release.[/green]")
                return True
            
            console.print(f"[cyan]Updating from {current_release} to {latest_release}...[/cyan]")
            
            # Create backup of index
            self._backup_index(current_release)
            
            # Process each dataset
            for dataset in self.datasets_to_download:
                success = self._update_dataset(dataset, current_release, latest_release)
                if not success:
                    self._restore_index_backup(current_release)
                    return False
                
            # Update successful - remove backup
            self._cleanup_backup(current_release)
            return True
            
        except Exception as e:
            console.print(f"[red]Error updating datasets: {str(e)}[/red]")
            self._restore_index_backup(current_release)
            return False

    def _update_dataset(self, dataset_name: str, current_release: str, target_release: str) -> bool:
        """Update a single dataset using diffs."""
        try:
            # Get diffs
            diff_url = f"{BASE_URL}/diffs/{current_release}/to/{target_release}/{dataset_name}"
            response = self.make_request(diff_url)
            diffs = response.json()
            
            dataset_dir = self.base_dir / target_release / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            with Progress() as progress:
                task = progress.add_task(f"Updating {dataset_name}...", total=len(diffs['diffs']))
                
                # Process each diff sequentially
                for diff in diffs['diffs']:
                    # Apply updates
                    for url in diff['update_files']:
                        temp_file = dataset_dir / f"temp_{Path(url).name}"
                        try:
                            self.download_file(url, temp_file)
                            self._update_index_for_file(temp_file, dataset_name, diff['to_release'])
                        finally:
                            if temp_file.exists():
                                temp_file.unlink()
                    
                    # Process deletes
                    for url in diff['delete_files']:
                        temp_file = dataset_dir / f"temp_delete_{Path(url).name}"
                        try:
                            self.download_file(url, temp_file)
                            self._remove_from_index(temp_file, dataset_name)
                        finally:
                            if temp_file.exists():
                                temp_file.unlink()
                            
                    progress.advance(task)
                
            return True
            
        except Exception as e:
            console.print(f"[red]Error updating dataset {dataset_name}: {str(e)}[/red]")
            return False

    def _backup_index(self, release_id: str):
        """Create backup of current index."""
        index_path = self.index_dir / f"{release_id}.db"
        backup_path = self.index_dir / f"{release_id}.db.bak"
        
        if index_path.exists():
            shutil.copy2(str(index_path), str(backup_path))
            console.print("[green]Created index backup[/green]")

    def _restore_index_backup(self, release_id: str):
        """Restore index from backup if update failed."""
        index_path = self.index_dir / f"{release_id}.db"
        backup_path = self.index_dir / f"{release_id}.db.bak"
        
        if backup_path.exists():
            if index_path.exists():
                index_path.unlink()
            shutil.copy2(str(backup_path), str(index_path))
            console.print("[yellow]Restored index from backup[/yellow]")

    def _cleanup_backup(self, release_id: str):
        """Remove backup files after successful update."""
        backup_path = self.index_dir / f"{release_id}.db.bak"
        if backup_path.exists():
            backup_path.unlink()

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
        index_path = self.index_dir / f"{self.current_release}.db"
        
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

    def verify_index_completeness(self) -> bool:
        """
        Verify that all downloaded files have been completely indexed.
        Returns True if all files are properly indexed, False otherwise.
        """
        try:
            release_id = self._get_latest_local_release()
            if not release_id:
                console.print("[yellow]No local datasets found.[/yellow]")
                return False

            index_path = self.index_dir / f"{release_id}.db"
            if not index_path.exists():
                console.print("[red]Index database not found.[/red]")
                return False

            incomplete_files = []
            
            with sqlite3.connect(str(index_path)) as conn:
                for dataset in self.datasets_to_download:
                    dataset_dir = self.base_dir / release_id / dataset
                    if not dataset_dir.exists():
                        continue

                    # Get all JSON files in the dataset directory
                    json_files = list(dataset_dir.glob('*.json'))
                    for file_path in json_files:
                        if file_path.name == 'metadata.json':
                            continue
                            
                        console.print(f"[cyan]Verifying index for {file_path.name}...[/cyan]")
                        
                        # Count actual records in the file
                        actual_records = 0
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    json.loads(line.strip())  # Validate JSON
                                    actual_records += 1
                                except json.JSONDecodeError:
                                    continue
                        
                        # Count indexed records for this file
                        cursor = conn.execute("""
                            SELECT COUNT(*) FROM paper_locations 
                            WHERE file_path = ?
                        """, (str(file_path),))
                        indexed_records = cursor.fetchone()[0]
                        
                        if indexed_records == 0:
                            console.print(f"[red]File {file_path.name} has no index entries![/red]")
                            incomplete_files.append((file_path, 'missing'))
                        elif indexed_records < actual_records:
                            console.print(
                                f"[yellow]File {file_path.name} is partially indexed: "
                                f"{indexed_records}/{actual_records} records[/yellow]"
                            )
                            incomplete_files.append((file_path, 'partial'))

            if incomplete_files:
                console.print("\n[red]Found incompletely indexed files:[/red]")
                for file_path, status in incomplete_files:
                    console.print(f"- {file_path.name} ({status})")
                
                # Offer to fix incomplete files
                if console.input("\nWould you like to reindex these files? (y/n): ").lower() == 'y':
                    with sqlite3.connect(str(index_path)) as conn:
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
                
                return False
                
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download Semantic Scholar datasets')
    parser.add_argument('--release', default='latest', help='Release ID to download')
    parser.add_argument('--mini', action='store_true', help='Download minimal dataset for testing')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded datasets')
    parser.add_argument('--verify-index', action='store_true', help='Verify index completeness')
    args = parser.parse_args()
    
    downloader = S2DatasetDownloader()
    
    if args.verify_index:
        downloader.verify_index_completeness()
    elif args.verify:
        downloader.verify_downloads(args.release)
    else:
        downloader.download_all_datasets(args.release, args.mini)

if __name__ == "__main__":
    main() 