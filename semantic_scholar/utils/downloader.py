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

    def make_request(self, url: str, method: str = 'get', max_retries: int = 5, **kwargs) -> requests.Response:
        """Make a request with retry logic for rate limits."""
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

                if response.status_code == 429:
                    wait_time = min(30, (2 ** attempt) + 1)  # Exponential backoff
                    console.print(f"[yellow]Rate limited. Waiting {wait_time} seconds...[/yellow]")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = min(30, (2 ** attempt) + 1)
                console.print(f"[yellow]Request failed. Retrying in {wait_time} seconds...[/yellow]")
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

    def download_file(self, url: str, output_dir: Path, desc: str = None) -> bool:
        """Download a file with progress bar."""
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
                
            # If file is gzipped, extract it
            if filename.endswith('.gz'):
                console.print(f"Extracting {filename}...")
                base_name = filename.replace('.gz', '')
                if not base_name.endswith('.json'):
                    base_name += '.json'
                
                with gzip.open(output_path, 'rt', encoding='utf-8') as f_in:
                    with open(output_dir / base_name, 'w', encoding='utf-8') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(output_path)  # Remove the gzipped file
                
            return True
            
        except Exception as e:
            console.print(f"[red]Error downloading {url}: {str(e)}[/red]")
            return False

    def _init_sqlite_db(self, index_path: Path):
        """Initialize SQLite database with proper schema and indices."""
        with sqlite3.connect(str(index_path)) as conn:
            # Enable WAL mode for better write performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
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

    def _index_file(self, conn: sqlite3.Connection, file_path: Path, dataset: str):
        """Build index for a downloaded file with optimized batch processing."""
        console.print(f"[cyan]Indexing {file_path.name}...[/cyan]")
        
        # Define which IDs to index for each dataset
        dataset_id_fields = {
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

        id_fields = dataset_id_fields.get(dataset, [])
        if not id_fields:
            console.print(f"[yellow]Warning: No ID fields defined for dataset {dataset}[/yellow]")
            return

        try:
            # Prepare the insert statement once
            insert_stmt = """
                INSERT OR REPLACE INTO paper_locations 
                (id, id_type, dataset, file_path, line_offset)
                VALUES (?, ?, ?, ?, ?)
            """
            
            # Create a batch of records to insert
            batch_size = 10000
            batch = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                offset = 0
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        
                        # Index all relevant IDs for this dataset
                        for field_name, id_type in id_fields:
                            id_value = str(item.get(field_name, '')).lower()
                            if id_value:
                                batch.append((
                                    id_value, 
                                    id_type, 
                                    dataset, 
                                    str(file_path), 
                                    offset
                                ))
                                
                    except json.JSONDecodeError:
                        console.print(f"[yellow]Warning: Invalid JSON at line {line_num}[/yellow]")
                        
                    offset += len(line.encode('utf-8'))
                    
                    # Execute batch insert when batch is full
                    if len(batch) >= batch_size:
                        conn.executemany(insert_stmt, batch)
                        batch = []
                        console.print(f"[green]Indexed {line_num} lines[/green]")
                
                # Insert any remaining records
                if batch:
                    conn.executemany(insert_stmt, batch)
                    console.print(f"[green]Indexed {line_num} lines[/green]")
                    
        except Exception as e:
            console.print(f"[red]Error indexing {file_path.name}: {str(e)}[/red]")

    def download_dataset(self, dataset_name: str, release_id: str = 'latest', mini: bool = False) -> bool:
        """Download a specific dataset and build index."""
        if release_id == 'latest':
            release_id = self.get_latest_release()
        
        try:
            dataset_info = self.get_dataset_info(dataset_name, release_id)
            if not dataset_info:
                return False
            
            dataset_dir = self.base_dir / release_id / dataset_name
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save metadata
            with open(dataset_dir / 'metadata.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)

            # Initialize SQLite index
            index_dir = self.base_dir / "indices"
            index_dir.mkdir(exist_ok=True)
            index_path = index_dir / f"{release_id}.db"
            
            # Initialize database schema if needed
            self._init_sqlite_db(index_path)
            
            with sqlite3.connect(str(index_path)) as conn:
                # Handle S2ORC differently
                if dataset_name == 's2orc':
                    files = dataset_info['files']
                    if mini:
                        files = files[:1]  # Get only first shard for mini download
                    
                    with Progress() as progress:
                        task = progress.add_task(f"Downloading {dataset_name}...", total=len(files))
                        
                        for file_info in files:
                            url = file_info['url']
                            shard = file_info['shard']
                            output_path = dataset_dir / f"{shard}.json"
                            
                            if self.download_file(url, dataset_dir, f"Downloading S2ORC shard {shard}"):
                                self._index_file(conn, output_path, dataset_name)
                            progress.advance(task)
                else:
                    # Standard dataset handling
                    files_to_download = dataset_info['files'][:1] if mini else dataset_info['files']
                    
                    with Progress() as progress:
                        task = progress.add_task(f"Downloading {dataset_name}...", total=len(files_to_download))
                        
                        for file_url in files_to_download:
                            output_path = dataset_dir / self.get_filename_from_url(file_url).replace('.gz', '.json')
                            if self.download_file(file_url, dataset_dir):
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download Semantic Scholar datasets')
    parser.add_argument('--release', default='latest', help='Release ID to download')
    parser.add_argument('--mini', action='store_true', help='Download minimal dataset for testing')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded datasets')
    args = parser.parse_args()
    
    downloader = S2DatasetDownloader()
    
    if args.verify:
        downloader.verify_downloads(args.release)
    else:
        downloader.download_all_datasets(args.release, args.mini)

if __name__ == "__main__":
    main() 