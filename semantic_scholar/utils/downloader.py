import os
import json
import requests
from typing import List, Dict, Optional, Tuple, Set
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Now we can import using the full package path
from semantic_scholar.utils.binary_indexer import BinaryIndexer, IndexEntry
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
    def __init__(self, version: Optional[str] = None):
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
        
        # Store the requested version
        self.version = version
        
        self.datasets_to_download = [
            "papers", 
            "abstracts",
            "authors",
            "s2orc",
            "tldrs"
        ]
        
        # Create base and index directories with parents
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.base_dir / "binary_indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Define which IDs to index for each dataset
        self.dataset_id_fields = {
            'papers': [
                ('paperId', 'paper_id'),
                ('corpusid', 'corpus_id')
            ],
            'abstracts': [('corpusid', 'corpus_id')],
            's2orc': [('corpusid', 'corpus_id')],
            'authors': [('authorid', 'author_id')],
            'tldrs': [('corpusid', 'corpus_id')]
        }
        
        # Initialize binary indexer
        self.indexer = BinaryIndexer(self.base_dir)
        
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
        """Get the latest release ID or return specified version."""
        if self.version:
            # Validate version format (YYYY-MM-DD)
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', self.version):
                raise ValueError("Version must be in YYYY-MM-DD format")
            return self.version
        
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
                    print("s2orc files")
                    print(data['files'])
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
                    print("Shard is ", file_info['shard'])
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
                    self.index_dataset(dataset_name, release_id)
                
            return True
                
        except Exception as e:
            console.print(f"[red]Error downloading dataset {dataset_name}: {str(e)}[/red]")
            return False

    def index_dataset(self, dataset: str, release_id: str, skip_ids: Optional[Set[str]] = None) -> bool:
        """Create binary indices for a dataset"""
        try:
            dataset_dir = self.base_dir / release_id / dataset
            if not dataset_dir.exists():
                console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
                return False

            # Get all data files
            files = [f for f in dataset_dir.glob("*.json") if f.name != 'metadata.json']
            if not files:
                console.print(f"[yellow]No files found to index in {dataset_dir}[/yellow]")
                return False

            # Create temporary directory for chunks
            chunk_dir = self.indexer.tmp_dir / f"{release_id}_{dataset}_chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Track entries for each ID type
            id_types_seen = set()
            total_entries = 0
            
            console.print(f"[cyan]Processing {len(files)} files for {dataset}...[/cyan]")
            
            # Process each file and write sorted chunks
            for file_num, file_path in enumerate(files):
                console.print(f"[cyan]Processing {file_path.name}...[/cyan]")
                entries_by_id_type = defaultdict(list)
                entries_in_file = 0
                
                with open(file_path, 'rb') as f:
                    offset = 0
                    for line in f:
                        try:
                            # Try hex-encoded JSON first
                            decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                            data = json.loads(decoded)
                        except:
                            # Fall back to regular JSON
                            try:
                                data = json.loads(line.strip())
                            except:
                                console.print(f"[yellow]Warning: Skipping invalid JSON line in {file_path}[/yellow]")
                                offset += len(line)
                                continue

                        # Extract IDs based on dataset type
                        if dataset == 'papers':
                            if 'corpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['corpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                            if 'paperId' in data:
                                entries_by_id_type['paper_id'].append(
                                    IndexEntry(data['paperId'], str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'authors':
                            if 'authorid' in data:
                                entries_by_id_type['author_id'].append(
                                    IndexEntry(data['authorid'], str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'citations':
                            if 'citingcorpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['citingcorpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                            if 'citedcorpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['citedcorpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'abstracts':
                            if 'corpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['corpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 's2orc':
                            if 'corpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['corpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                        elif dataset == 'tldrs':
                            if 'corpusid' in data:
                                entries_by_id_type['corpus_id'].append(
                                    IndexEntry(str(data['corpusid']), str(file_path), offset)
                                )
                                entries_in_file += 1
                            
                        offset += len(line)

                # Write sorted chunks for each ID type
                for id_type, entries in entries_by_id_type.items():
                    id_types_seen.add(id_type)
                    entries.sort(key=lambda x: x.id)
                    chunk_path = chunk_dir / f"{id_type}_chunk_{file_num:03d}.idx"

                    # Skip rewriting if chunk file already exists
                    if chunk_path.exists():
                        console.print(f"[yellow]Skipping existing chunk {chunk_path.name}[/yellow]")
                        continue

                    with open(chunk_path, 'wb') as f:
                        for entry in entries:
                            if skip_ids is None or entry.id not in skip_ids:
                                f.write(entry.to_bytes())

                total_entries += entries_in_file
                console.print(f"[green]Created {entries_in_file:,} index entries from {file_path.name}[/green]")

            # Create final indices from chunks
            for id_type in id_types_seen:
                chunks = sorted(chunk_dir.glob(f"{id_type}_chunk_*.idx"))
                if not chunks:
                    continue
                    
                if not self.indexer.create_index_from_chunks(release_id, dataset, id_type, chunks):
                    console.print(f"[red]Failed to create index for {dataset}_{id_type}[/red]")
                    return False

            # Comment out or remove the cleanup:
            # shutil.rmtree(chunk_dir)

            console.print(f"[green]Successfully created {total_entries:,} total index entries for {dataset}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error indexing dataset {dataset}: {str(e)}[/red]")
            if 'chunk_dir' in locals() and chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            return False

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indexer.close()

    def download_all_datasets(self, release_id: str = 'latest', mini: bool = False):
        """Download all datasets first, then index them all."""
        if release_id == 'latest' and self.version:
            release_id = self.version
        elif release_id == 'latest':
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
        """Incrementally update all datasets to the latest release using the official
        diff end-points provided by Semantic Scholar.  If the diff API is not
        available (or something goes wrong mid-way) we gracefully fall back to
        downloading the full dataset for the new release – this guarantees that
        Valsci will continue to work even on the first update attempt.

        The high-level algorithm is:

        1.  Identify the *current* (local) release as well as the *latest*
            release available from the API.
        2.  For each dataset:
            a.  Copy the existing JSONL shard files from the current release to
                a new directory for the latest release.  Hard-links are used on
                filesystems that support them to save disk space; otherwise a
                normal copy is performed.
            b.  Download every «update» diff file and place it in the new
                directory.
            c.  Collect the primary-key values that appear in «delete» diff
                files so we can exclude them when (re)building the binary index
                for the new release.
        3.  Re-index the dataset (calling the enhanced index_dataset method
            which accepts a *skip_ids* set).
        4.  Run a quick verification step to ensure the rebuilt index is
            internally consistent with the source files.

        Returns True on success, False otherwise.
        """

        try:
            current_release = self._get_latest_local_release()
            if not current_release:
                console.print("[yellow]No local datasets found. Please run an initial download first.[/yellow]")
                return False

            latest_release = self.get_latest_release()
            if current_release == latest_release:
                console.print("[green]Datasets already at the latest release – nothing to do.[/green]")
                return True

            console.print(f"[cyan]Updating from {current_release} ➜ {latest_release} using diffs...[/cyan]")

            for dataset in self.datasets_to_download:
                console.print(f"\n[bold]Dataset: {dataset}[/bold]")

                # We will lazily create/populate the destination directory *only*
                # if there are any diff chunks to apply.  This avoids copying a
                # potentially huge dataset when we subsequently fall back to a
                # full download.
                dst_dir: Optional[Path] = None
                deletion_ids: Set[str] = set()

                # ------------------------------------------------------------------
                # Retrieve diff manifest for this dataset
                # ------------------------------------------------------------------
                if dataset == 's2orc':
                    diff_url = f"https://api.semanticscholar.org/datasets/v1/diffs/{current_release}/to/{latest_release}/s2orc/"
                else:
                    diff_url = f"{BASE_URL}/diffs/{current_release}/to/{latest_release}/{dataset}"

                try:
                    diffs_resp = self.make_request(diff_url)
                    diffs = diffs_resp.json().get('diffs', [])
                except Exception as e:
                    console.print(f"[yellow]Could not retrieve diff for {dataset} (reason: {e}). Falling back to full download.[/yellow]")
                    # Full download fall-back
                    self.download_dataset(dataset, latest_release, mini=False, index=True)
                    continue

                total_diff_files = sum(len(d.get('update_files', [])) + len(d.get('delete_files', [])) for d in diffs)
                if total_diff_files == 0:
                    console.print("[green]No changes for this dataset – skipping.[/green]")
                    # Even if there are no changes we still need an index file
                    # for the new release so we copy the existing one.
                    for field, id_type in self.dataset_id_fields[dataset]:
                        old_idx = self.index_dir / f"{current_release}_{dataset}_{id_type}.idx"
                        new_idx = self.index_dir / f"{latest_release}_{dataset}_{id_type}.idx"
                        if old_idx.exists() and not new_idx.exists():
                            shutil.copy2(old_idx, new_idx)
                    continue

                # At this point we know there are some diff files – now we can
                # prepare the destination directory by copying (or hard-linking)
                # the current release.
                dst_dir = self._prepare_dataset_dir_for_update(current_release, latest_release, dataset)

                with Progress() as progress:
                    task_id = progress.add_task(f"[cyan]Applying diffs for {dataset}...", total=total_diff_files)

                    # Loop through every diff segment chronologically.
                    for diff in diffs:
                        # ---------------------------
                        #   UPDATE  files
                        # ---------------------------
                        upd_urls = diff.get('update_files', [])
                        if dataset == 's2orc':
                            upd_urls = [u['url'] for u in upd_urls]

                        for url in upd_urls:
                            success, _ = self.download_file(url, dst_dir)
                            if not success:
                                console.print(f"[red]Failed to download update file: {url}[/red]")
                            progress.advance(task_id)

                        # ---------------------------
                        #   DELETE files
                        # ---------------------------
                        del_urls = diff.get('delete_files', [])
                        if dataset == 's2orc':
                            del_urls = [u['url'] for u in del_urls]

                        for url in del_urls:
                            success, del_path = self.download_file(url, dst_dir)
                            if success and del_path:
                                # Extract primary keys that need to be removed
                                try:
                                    with open(del_path, 'rb') as f_del:
                                        for line in f_del:
                                            try:
                                                decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                                                rec = json.loads(decoded)
                                            except Exception:
                                                try:
                                                    rec = json.loads(line.strip())
                                                except Exception:
                                                    continue

                                            for field, _ in self.dataset_id_fields[dataset]:
                                                if field in rec:
                                                    deletion_ids.add(str(rec[field]))
                                finally:
                                    # Delete the diff delete file – we only
                                    # needed it to collect the IDs.
                                    del_path.unlink(missing_ok=True)
                            progress.advance(task_id)

                # After all diff chunks processed, rebuild the binary indices
                console.print("[cyan]Re-building binary index...[/cyan]")
                if not self.index_dataset(dataset, latest_release, skip_ids=deletion_ids):
                    console.print(f"[red]Failed to rebuild index for {dataset}")
                    return False

                # Quick verification (±10 % margin).
                self.verify_index_completeness(latest_release, dataset, quick_estimate=True)

            console.print("\n[green]All datasets updated successfully![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error during incremental update: {e}")
            return False

    def _prepare_dataset_dir_for_update(self, current_release: str, latest_release: str, dataset: str) -> Path:
        """Helper that copies (or hard-links) the current-release dataset files
        into a fresh directory for the *latest_release*.  Returns the new
        directory path.  If the directory already exists it is returned as-is.
        """

        src_dir = self.base_dir / current_release / dataset
        dst_dir = self.base_dir / latest_release / dataset

        if dst_dir.exists():
            return dst_dir

        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            # This should not happen but we guard against it.
            return dst_dir

        for file in src_dir.glob("*.json"):
            # Skip metadata.json here; we'll handle it explicitly below
            if file.name == "metadata.json":
                continue

            dest = dst_dir / file.name
            if dest.exists():
                # File (or hard-link) already in place
                continue
            try:
                os.link(file, dest)  # Hard-link (O(1) if same filesystem)
            except Exception:
                shutil.copy2(file, dest)  # Fall-back to regular copy

        # Copy metadata.json once (if it isn't already present)
        meta_src = src_dir / "metadata.json"
        meta_dst = dst_dir / "metadata.json"
        if meta_src.exists() and not meta_dst.exists():
            try:
                os.link(meta_src, meta_dst)
            except Exception:
                shutil.copy2(meta_src, meta_dst)

        return dst_dir

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
        
        # Get index stats from binary indexer
        index_stats = self.indexer.get_index_stats(release_id)
        if not index_stats:
            console.print("[red]No index data found for this release[/red]")
            return
        
        table = Table(
            "Dataset", 
            "Expected Files", 
            "Downloaded Files",
            "Index Status",
            title=f"Dataset Audit for Release {release_id}"
        )
        
        for dataset in datasets:
            try:
                # Get expected files from API
                dataset_info = self.get_dataset_info(dataset, release_id)
                if not dataset_info:
                    table.add_row(
                        dataset,
                        "?",
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
                
                # Get index status for all ID types for this dataset
                index_statuses = []
                for field_name, id_type in self.dataset_id_fields[dataset]:
                    index_key = f"{dataset}_{id_type}"
                    if index_key in index_stats:
                        index_statuses.append(
                            f"{id_type}: {index_stats[index_key]['entry_count']:,}"
                        )
                
                if index_statuses:
                    index_status = "[green]" + "\n".join(index_statuses) + "[/green]"
                else:
                    index_status = "[yellow]Not indexed[/yellow]"
                
                table.add_row(
                    dataset,
                    str(expected_count),
                    str(len(downloaded_files)),
                    index_status
                )
                
            except Exception as e:
                table.add_row(
                    dataset,
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
        
        # Get index stats from binary indexer
        stats = self.indexer.get_index_stats(release_id)
        if not stats:
            console.print("[red]No index data found for this release[/red]")
            return
        
        table = Table(
            "Dataset",
            "ID Type",
            "Entries",
            "Size",
            "Created",
            "Status",
            title=f"Index Statistics for Release {release_id}"
        )
        
        total_entries = 0
        for index_key, info in stats.items():
            dataset, id_type = index_key.split('_', 1)
            total_entries += info['entry_count']
            
            status = "[green]Healthy[/green]" if info['healthy'] else "[red]Unhealthy[/red]"
            
            table.add_row(
                dataset,
                id_type,
                f"{info['entry_count']:,}",
                f"{info['size_mb']:.1f} MB",
                info['created'],
                status
            )
        
        console.print(f"\nTotal index entries: [bold cyan]{total_entries:,}[/bold cyan]\n")
        console.print(table)

    def _verify_db_name(self):
        """This method is no longer needed with binary indexer"""
        pass

    def verify_index_completeness(self, release_id: str, dataset: Optional[str] = None, 
                                sample_size: int = 1000, quick_estimate: bool = False) -> bool:
        """Thin wrapper around BinaryIndexer.verify_index_completeness so callers
        don't need to import the indexer directly.
        """
        try:
            return self.indexer.verify_index_completeness(release_id, dataset, quick_estimate=quick_estimate)
        except Exception as e:
            console.print(f"[red]Error verifying index completeness: {str(e)}[/red]")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download Semantic Scholar datasets')
    parser.add_argument('--release', default='latest', help='Release ID to download')
    parser.add_argument('--version', help='Specific version to download (YYYY-MM-DD format)')
    parser.add_argument('--mini', action='store_true', help='Download minimal dataset for testing')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded datasets')
    parser.add_argument('--verify-index', nargs='*', help='Verify index completeness. Optionally specify datasets to verify')
    parser.add_argument('--audit', nargs='*', help='Audit datasets and indexing status')
    parser.add_argument('--index-only', nargs='*', help='Only run indexing on downloaded files')
    parser.add_argument('--repair', action='store_true', help='Repair/resume incomplete indexes')
    parser.add_argument('--count', action='store_true', help='Show detailed index counts for each file')
    # New flag to perform an incremental update based on diff end-points
    parser.add_argument('--update', action='store_true', help='Incrementally update all local datasets to the latest release using Semantic Scholar diff files')
    args = parser.parse_args()
    
    with S2DatasetDownloader(version=args.version) as downloader:
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

        if args.verify:
            # Verify downloaded files match expected files from API
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader.get_latest_release()
            
            console.print(f"\n[bold cyan]Verifying downloads for release {release_id}...[/bold cyan]")
            missing_files = {}
            
            for dataset in downloader.datasets_to_download:
                try:
                    dataset_info = downloader.get_dataset_info(dataset, release_id)
                    if not dataset_info:
                        console.print(f"[yellow]Could not get info for dataset: {dataset}[/yellow]")
                        continue
                        
                    dataset_dir = downloader.base_dir / release_id / dataset
                    if not dataset_dir.exists():
                        missing_files[dataset] = ["entire dataset missing"]
                        continue
                    
                    # Get expected files
                    if dataset == 's2orc':
                        expected_files = [
                            f"{info['shard']}.json" 
                            for info in (dataset_info['files'][:1] if args.mini else dataset_info['files'])
                        ]
                    else:
                        expected_files = [
                            downloader.get_filename_from_url(url).replace('.gz', '.json')
                            for url in (dataset_info['files'][:1] if args.mini else dataset_info['files'])
                        ]
                    
                    # Check actual files
                    actual_files = {f.name for f in dataset_dir.glob('*.json') if f.name != 'metadata.json'}
                    missing = set(expected_files) - actual_files
                    if missing:
                        missing_files[dataset] = missing
                        
                except Exception as e:
                    console.print(f"[red]Error verifying {dataset}: {str(e)}[/red]")
            
            if missing_files:
                console.print("\n[red]Missing files found:[/red]")
                for dataset, files in missing_files.items():
                    console.print(f"\n[yellow]{dataset}:[/yellow]")
                    for f in files:
                        console.print(f"  • {f}")
                return False
            else:
                console.print("\n[green]All expected files are present![/green]")
                return True
                
        elif args.count:
            # Show index statistics
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return
                
            stats = downloader.indexer.get_index_stats(release_id)
            
            # Create a rich table to display stats
            table = Table(title=f"Index Statistics for Release {release_id}")
            table.add_column("Dataset")
            table.add_column("ID Type")
            table.add_column("Entries")
            table.add_column("Size")
            table.add_column("Created")
            table.add_column("Status")
            
            for index_name, info in stats.items():
                dataset, id_type = index_name.split('_')
                status = "[green]Healthy[/green]" if info['healthy'] else "[red]Unhealthy[/red]"
                table.add_row(
                    dataset,
                    id_type,
                    f"{info['entry_count']:,}",
                    f"{info['size_mb']:.1f} MB",
                    info['created'],
                    status
                )
            
            console.print(table)
            
        elif args.audit is not None:
            datasets = validate_datasets(args.audit)
            if datasets is None:
                return
                
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return
                
            # Create audit table
            table = Table(title=f"Dataset Audit for Release {release_id}")
            table.add_column("Dataset")
            table.add_column("Files")
            table.add_column("Index Status")
            table.add_column("Health Check")
            
            for dataset in datasets:
                # Check dataset files
                dataset_dir = downloader.base_dir / release_id / dataset
                if not dataset_dir.exists():
                    table.add_row(dataset, "[red]Missing[/red]", "N/A", "N/A")
                    continue
                    
                files = list(dataset_dir.glob("*.json"))
                file_count = len([f for f in files if f.name != 'metadata.json'])
                
                # Get index stats
                stats = downloader.indexer.get_index_stats(release_id)
                index_info = next((v for k, v in stats.items() if k.startswith(f"{dataset}_")), None)
                
                if not index_info:
                    table.add_row(
                        dataset,
                        f"{file_count} files",
                        "[yellow]Not Indexed[/yellow]",
                        "N/A"
                    )
                else:
                    health = "[green]Healthy[/green]" if index_info['healthy'] else "[red]Unhealthy[/red]"
                    table.add_row(
                        dataset,
                        f"{file_count} files",
                        f"{index_info['entry_count']:,} entries",
                        health
                    )
            
            console.print(table)
            
        elif args.verify_index is not None:
            datasets = validate_datasets(args.verify_index)
            if datasets is None:
                return
                
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return
                
            console.print(f"[cyan]Verifying indices for {len(datasets)} datasets...[/cyan]")
            
            # First do a quick estimate check
            console.print(f"\n[bold]Quick estimation check...[/bold]")
            if downloader.verify_index_completeness(release_id, quick_estimate=True):
                console.print(f"[green]✓ Quick estimate check passed[/green]")
                
                # If quick check passes, offer to do detailed verification
                console.print("\nQuick check passed. Would you like to perform a detailed verification? (y/N)")
                response = input().lower()
                if response == 'y':
                    console.print(f"\n[bold]Performing detailed verification...[/bold]")
                    if downloader.indexer.verify_all_indices(release_id, show_details=True):
                        console.print(f"[green]✓ All indices verified successfully[/green]")
                    else:
                        console.print(f"[red]× Indices verification failed[/red]")
            else:
                console.print(f"[red]× Quick estimate check failed[/red]")
                console.print("\nWould you like to perform a detailed verification to identify issues? (y/N)")
                response = input().lower()
                if response == 'y':
                    console.print(f"\n[bold]Performing detailed verification...[/bold]")
                    downloader.indexer.verify_all_indices(release_id, show_details=True)
                    
        elif args.update:
            # Perform an incremental update (diff-based). This downloads only the changes
            # between the current local release and the latest available release, then
            # rebuilds the binary indices.  Falls back to a full download if diffs are
            # unavailable.
            success = downloader.update_datasets()
            # Exit code indicates success (0) or failure (1) for shell scripts
            sys.exit(0 if success else 1)
        elif args.index_only is not None:
            datasets = validate_datasets(args.index_only)
            if datasets is None:
                return
                
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()
            if not release_id:
                console.print("[red]No local releases found[/red]")
                return
                
            console.print(f"[cyan]Indexing {len(datasets)} datasets...[/cyan]")
            
            for dataset in datasets:
                console.print(f"\n[bold]Indexing {dataset}...[/bold]")
                if downloader.index_dataset(dataset, release_id):
                    console.print(f"[green]✓ Successfully indexed {dataset}[/green]")
                else:
                    console.print(f"[red]× Failed to index {dataset}[/red]")
                    
        elif args.repair:
            # Repair mode: re-index datasets that are missing or unhealthy for the latest local release
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader._get_latest_local_release()

            if not release_id:
                console.print("[red]No local releases found to repair[/red]")
                return

            console.print(f"[bold cyan]Repairing indices for release {release_id}...[/bold cyan]")

            # Get current index stats (may be empty)
            stats = downloader.indexer.get_index_stats(release_id)

            for dataset in downloader.datasets_to_download:
                # Determine for every id_type if index exists and is healthy
                needs_rebuild = False
                for _, id_type in downloader.dataset_id_fields[dataset]:
                    key = f"{dataset}_{id_type}"
                    if key not in stats:
                        needs_rebuild = True
                        break
                    if not stats[key]['healthy']:
                        needs_rebuild = True
                        break

                if needs_rebuild:
                    console.print(f"\n[bold]Re-indexing {dataset}...[/bold]")
                    if downloader.index_dataset(dataset, release_id):
                        console.print(f"[green]✓ Successfully re-indexed {dataset}[/green]")
                    else:
                        console.print(f"[red]× Failed to re-index {dataset}[/red]")

            console.print("\n[bold cyan]Repair completed[/bold cyan]")

        else:
            # Download and index all datasets
            release_id = args.release
            if release_id == 'latest':
                release_id = downloader.get_latest_release()
                
            console.print(f"[bold cyan]Downloading and indexing datasets for release {release_id}...[/bold cyan]")
            
            for dataset in downloader.datasets_to_download:
                console.print(f"\n[bold]Processing {dataset}...[/bold]")
                if downloader.download_dataset(dataset, release_id, args.mini, index=True):
                    console.print(f"[green]✓ Successfully processed {dataset}[/green]")
                else:
                    console.print(f"[red]× Failed to process {dataset}[/red]")

if __name__ == "__main__":
    main() 