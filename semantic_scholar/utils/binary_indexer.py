import os
import json
import mmap
import struct
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from rich.console import Console
import hashlib
import tempfile
from datetime import datetime
from collections import defaultdict
import random
from rich.table import Table
import heapq
import logging
from itertools import count

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class IndexEntry:
    id: str  # Will be padded/truncated to 40 bytes
    file_path: str  # Will be padded/truncated to 256 bytes  
    offset: int  # 8 bytes
    
    ENTRY_FORMAT = '40s256sQ'  # Q = unsigned long long (8 bytes)
    ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'IndexEntry':
        id_bytes, path_bytes, offset = struct.unpack(cls.ENTRY_FORMAT, data)
        return cls(
            id=id_bytes.decode('utf-8').rstrip('\0'),
            file_path=path_bytes.decode('utf-8').rstrip('\0'),
            offset=offset
        )
    
    def to_bytes(self) -> bytes:
        id_bytes = self.id.encode('utf-8').ljust(40, b'\0')[:40]
        path_bytes = self.file_path.encode('utf-8').ljust(256, b'\0')[:256]
        return struct.pack(self.ENTRY_FORMAT, id_bytes, path_bytes, self.offset)

class BinaryIndexer:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.index_dir = self.base_dir / "binary_indices"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tmp directory
        self.tmp_dir = self.index_dir / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        
        # Track open memory maps
        self._mmaps: Dict[str, mmap.mmap] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Define which IDs to index for each dataset
        self.dataset_id_fields = {
            'papers': [('corpusid', 'corpus_id')],  # Remove paper_id entry
            'abstracts': [('corpusid', 'corpus_id')],
            's2orc': [('corpusid', 'corpus_id')],
            'authors': [('authorid', 'author_id')],
            'tldrs': [('corpusid', 'corpus_id')]
        }
        
    def close(self):
        """Close all open memory maps"""
        for mmap_obj in self._mmaps.values():
            mmap_obj.close()
        self._mmaps.clear()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_index_path(self, release_id: str, dataset: str, id_type: str) -> Path:
        """Get path for a specific index file"""
        path = self.index_dir / f"{release_id}_{dataset}_{id_type}.idx"
        logger.info(f"Constructing index path: {path}")
        if not path.exists():
            logger.warning(f"Index file does not exist: {path}")
        return path
        
    def _get_metadata_path(self, release_id: str) -> Path:
        """Get path for index metadata file"""
        return self.index_dir / f"{release_id}_metadata.json"

    def _load_metadata(self, release_id: str):
        """Load metadata for a release's indices"""
        path = self._get_metadata_path(release_id)
        if path.exists():
            with open(path) as f:
                self.metadata[release_id] = json.load(f)
        else:
            self.metadata[release_id] = {}

    def _save_metadata(self, release_id: str):
        """Save metadata for a release's indices"""
        path = self._get_metadata_path(release_id)
        with open(path, 'w') as f:
            json.dump(self.metadata[release_id], f, indent=2)

    def create_index(self, release_id: str, dataset: str, id_type: str, 
                    entries: List[IndexEntry], verify: bool = True) -> bool:
        """
        Create a new binary index file.
        Uses a temporary file and only replaces existing index if successful.
        """
        tmp_path = None
        try:
            console.print(f"[cyan]Creating index for {dataset}_{id_type}...[/cyan]")
            
            # Sort entries by ID for binary search
            console.print(f"[cyan]Sorting {len(entries):,} entries...[/cyan]")
            entries.sort(key=lambda x: x.id)
            
            # Create unique temporary file
            tmp_path = self.tmp_dir / f"{release_id}_{dataset}_{id_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.idx.tmp"
            final_path = self._get_index_path(release_id, dataset, id_type)

            # If index already exists and is valid, skip
            if final_path.exists():
                if release_id in self.metadata and f"{dataset}_{id_type}" in self.metadata[release_id]:
                    meta = self.metadata[release_id][f"{dataset}_{id_type}"]
                    if meta['entry_count'] == len(entries):
                        checksum = self._calculate_file_checksum(final_path)
                        if checksum == meta['checksum']:
                            console.print(f"[green]Index already exists and is valid: {final_path.name}[/green]")
                            return True

            # Write sorted entries to temporary file
            console.print(f"[cyan]Writing entries to temporary file {tmp_path.name}...[/cyan]")
            with open(tmp_path, 'wb') as f:
                for entry in entries:
                    f.write(entry.to_bytes())

            # Verify the temporary index
            if verify:
                console.print("[cyan]Verifying index integrity...[/cyan]")
                if not self._verify_index(tmp_path, entries):
                    raise ValueError("Index verification failed")

            # Calculate checksum
            console.print("[cyan]Calculating index checksum...[/cyan]")
            checksum = self._calculate_file_checksum(tmp_path)

            # Update metadata
            if release_id not in self.metadata:
                self._load_metadata(release_id)
            
            self.metadata[release_id][f"{dataset}_{id_type}"] = {
                'entry_count': len(entries),
                'checksum': checksum,
                'entry_size': IndexEntry.ENTRY_SIZE,
                'created': str(datetime.now())
            }
            
            # Move temporary file to final location
            console.print(f"[cyan]Moving index to final location: {final_path.name}...[/cyan]")
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(tmp_path), str(final_path))
            
            # Save updated metadata
            self._save_metadata(release_id)
            
            console.print(f"[green]Successfully created index with {len(entries):,} entries[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error creating index: {str(e)}[/red]")
            # Clean up temporary file if it exists
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return False

    def create_index_from_chunks(self, release_id: str, dataset: str, id_type: str, 
                               chunk_paths: List[Path], verify: bool = True) -> bool:
        """
        Create a new binary index file from sorted chunks.
        Uses a temporary file and only replaces existing index if successful.
        """
        tmp_path = None
        try:
            console.print(f"[cyan]Creating index for {dataset}_{id_type} from chunks...[/cyan]")
            
            # Create unique temporary file
            tmp_path = self.tmp_dir / f"{release_id}_{dataset}_{id_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.idx.tmp"
            final_path = self._get_index_path(release_id, dataset, id_type)

            # Merge chunks directly to temporary file
            entry_count = 0
            with open(tmp_path, 'wb') as out_f:
                # Initialize heap with first entry from each chunk
                heap = []
                chunk_files = []
                
                _tie_breaker = count()

                for chunk_path in chunk_paths:
                    f = open(chunk_path, 'rb')
                    chunk_files.append(f)
                    data = f.read(IndexEntry.ENTRY_SIZE)
                    if data:
                        entry = IndexEntry.from_bytes(data)
                        # Use a monotonically increasing tie-breaker to avoid comparing IndexEntry objects when IDs are identical
                        heap.append((entry.id, next(_tie_breaker), entry, f))
                
                heapq.heapify(heap)
                
                # Merge chunks
                while heap:
                    _, _, entry, chunk_f = heapq.heappop(heap)
                    out_f.write(entry.to_bytes())
                    entry_count += 1
                    
                    # Read next entry from this chunk
                    data = chunk_f.read(IndexEntry.ENTRY_SIZE)
                    if data:
                        next_entry = IndexEntry.from_bytes(data)
                        heapq.heappush(heap, (next_entry.id, next(_tie_breaker), next_entry, chunk_f))

            # Close all chunk files
            for f in chunk_files:
                f.close()

            # Calculate checksum
            console.print("[cyan]Calculating index checksum...[/cyan]")
            checksum = self._calculate_file_checksum(tmp_path)

            # Update metadata
            if release_id not in self.metadata:
                self._load_metadata(release_id)
            
            self.metadata[release_id][f"{dataset}_{id_type}"] = {
                'entry_count': entry_count,
                'checksum': checksum,
                'entry_size': IndexEntry.ENTRY_SIZE,
                'created': str(datetime.now())
            }
            
            # Move temporary file to final location
            console.print(f"[cyan]Moving index to final location: {final_path.name}...[/cyan]")
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(tmp_path), str(final_path))
            
            # Save updated metadata
            self._save_metadata(release_id)
            
            console.print(f"[green]Successfully created index with {entry_count:,} entries[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error creating index: {str(e)}[/red]")
            # Clean up temporary file if it exists
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            return False

    def _calculate_file_checksum(self, path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_index(self, path: Path, original_entries: List[IndexEntry]) -> bool:
        """Verify that an index file matches the original entries"""
        try:
            # Check file size
            expected_size = len(original_entries) * IndexEntry.ENTRY_SIZE
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                console.print(f"[red]Size mismatch: expected {expected_size}, got {actual_size}[/red]")
                return False

            # Read and verify each entry
            with open(path, 'rb') as f:
                for i, expected in enumerate(original_entries):
                    data = f.read(IndexEntry.ENTRY_SIZE)
                    entry = IndexEntry.from_bytes(data)
                    if entry.id != expected.id or entry.offset != expected.offset:
                        console.print(f"[red]Entry mismatch at position {i}[/red]")
                        return False

            return True

        except Exception as e:
            console.print(f"[red]Error verifying index: {str(e)}[/red]")
            return False

    def search(self, release_id: str, dataset: str, id_type: str, search_id: str) -> Optional[IndexEntry]:
        """
        Binary search for an ID in the index.
        Returns None if not found.
        """
        try:
            index_path = self._get_index_path(release_id, dataset, id_type)
            if not index_path.exists():
                return None

            # Get or create memory map
            mmap_key = f"{release_id}_{dataset}_{id_type}"
            if mmap_key not in self._mmaps:
                with open(index_path, 'rb') as f:
                    self._mmaps[mmap_key] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            mm = self._mmaps[mmap_key]
            entry_size = IndexEntry.ENTRY_SIZE
            
            # Binary search
            left = 0
            right = mm.size() // entry_size - 1
            
            while left <= right:
                mid = (left + right) // 2
                mm.seek(mid * entry_size)
                entry = IndexEntry.from_bytes(mm.read(entry_size))
                
                if entry.id == search_id:
                    return entry
                elif entry.id < search_id:
                    left = mid + 1
                else:
                    right = mid - 1
                    
            return None

        except Exception as e:
            console.print(f"[red]Error searching index: {str(e)}[/red]")
            return None

    def _verify_index_contents(self, index_path: Path, meta: dict) -> bool:
        """Verify that index file contains valid entries in correct format."""
        try:
            with open(index_path, 'rb') as f:
                # Check we can read the expected number of entries
                entry_count = meta['entry_count']
                valid_entries = 0
                prev_id = None  # For checking sorting
                
                # Read and validate each entry
                for i in range(entry_count):
                    try:
                        data = f.read(IndexEntry.ENTRY_SIZE)
                        if len(data) != IndexEntry.ENTRY_SIZE:
                            console.print(f"[red]Truncated entry at position {i}[/red]")
                            return False
                            
                        # Try to parse the entry
                        entry = IndexEntry.from_bytes(data)
                        
                        # Basic validation
                        if not entry.id or not entry.file_path or entry.offset < 0:
                            console.print(f"[red]Invalid entry at position {i}: {entry}[/red]")
                            return False
                            
                        # Check sorting (IDs should be in ascending order)
                        if prev_id and entry.id < prev_id:
                            console.print(f"[red]Index not properly sorted at position {i}[/red]")
                            return False
                        prev_id = entry.id
                        
                        # Verify referenced file exists
                        ref_file = Path(entry.file_path)
                        if not ref_file.exists():
                            console.print(f"[red]Referenced file missing: {ref_file}[/red]")
                            return False
                            
                        # Verify offset is within file bounds
                        if entry.offset >= ref_file.stat().st_size:
                            console.print(f"[red]Invalid offset {entry.offset} for file {ref_file}[/red]")
                            return False
                            
                        valid_entries += 1
                        
                    except struct.error:
                        console.print(f"[red]Failed to parse entry at position {i}[/red]")
                        return False
                        
                # Verify we found the expected number of valid entries
                if valid_entries != entry_count:
                    console.print(f"[red]Expected {entry_count} entries but found {valid_entries}[/red]")
                    return False
                    
                return True
                
        except Exception as e:
            console.print(f"[red]Error verifying index contents: {str(e)}[/red]")
            return False

    def verify_all_indices(self, release_id: str, show_details: bool = True) -> bool:
        """Verify all indices for a release, optionally showing more details."""
        try:
            self._load_metadata(release_id)
            if release_id not in self.metadata:
                console.print(f"[yellow]No metadata found for release {release_id}[/yellow]")
                return False

            # First check if any indices exist for this release
            index_files = list(self.index_dir.glob(f"{release_id}_*.idx"))
            if not index_files:
                console.print(f"[yellow]No index files found for release {release_id}[/yellow]")
                return False

            all_valid = True
            if show_details:
                console.print(f"[bold cyan]Verifying all indices for release {release_id}...[/bold cyan]")

            datasets_with_indices = set()

            # First verify index file integrity
            console.print("\n[bold]1. Verifying index file integrity...[/bold]")
            for index_path in index_files:
                try:
                    # Extract dataset and id_type from filename
                    filename = index_path.name
                    parts = filename[len(f"{release_id}_"):-4].split('_')
                    if len(parts) < 2:
                        continue
                    
                    dataset = parts[0]
                    id_type = '_'.join(parts[1:])
                    datasets_with_indices.add(dataset)
                    
                    index_key = f"{dataset}_{id_type}"
                    
                    if show_details:
                        console.print(f"[white]Checking index: [bold]{index_key}[/bold][/white]")

                    # Basic file checks
                    if not index_path.exists():
                        console.print(f"[red]Index file missing: {index_path}[/red]")
                        all_valid = False
                        continue

                    if release_id in self.metadata and index_key in self.metadata[release_id]:
                        meta = self.metadata[release_id][index_key]
                        
                        # Check file size
                        expected_size = meta['entry_count'] * IndexEntry.ENTRY_SIZE
                        actual_size = index_path.stat().st_size
                        if actual_size != expected_size:
                            console.print(
                                f"[red]Size mismatch for {index_key}: "
                                f"expected {expected_size}, got {actual_size}[/red]"
                            )
                            all_valid = False
                            continue

                        # Check checksum
                        current_checksum = self._calculate_file_checksum(index_path)
                        if current_checksum != meta['checksum']:
                            console.print(
                                f"[red]Checksum mismatch for {index_key}. "
                                f"Found {current_checksum}[/red]"
                            )
                            all_valid = False
                            continue
                            
                        # Verify actual index contents
                        if not self._verify_index_contents(index_path, meta):
                            console.print(f"[red]Index content verification failed for {index_key}[/red]")
                            all_valid = False
                            continue
                            
                    else:
                        console.print(f"[red]No metadata found for index {index_key}[/red]")
                        all_valid = False
                        continue

                    if show_details:
                        console.print(f"[green]Verified {index_key} successfully[/green]")

                except Exception as e:
                    console.print(f"[red]Error verifying index {index_path.name}: {str(e)}[/red]")
                    all_valid = False

            # Second, verify completeness against source files
            console.print("\n[bold]2. Verifying index completeness against source files...[/bold]")
            if not self.verify_index_completeness(release_id):
                all_valid = False

            # Finally, report datasets without indices
            console.print("\n[bold]3. Checking for missing dataset indices...[/bold]")
            for dataset in ['papers', 'abstracts', 'citations', 'authors', 's2orc', 'tldrs']:
                if dataset not in datasets_with_indices:
                    console.print(f"[yellow]No indices found for dataset: {dataset}[/yellow]")
                    all_valid = False

            if all_valid:
                console.print("\n[green]✓ All verification checks passed successfully[/green]")
            else:
                console.print("\n[red]× Some verification checks failed[/red]")

            return all_valid

        except Exception as e:
            console.print(f"[red]Error verifying indices: {str(e)}[/red]")
            return False

    def __del__(self):
        """Ensure cleanup of resources"""
        self.close()
        
        # Clean up temporary directory
        if hasattr(self, 'tmp_dir') and self.tmp_dir.exists():
            try:
                shutil.rmtree(self.tmp_dir)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not clean up temp directory: {e}[/yellow]")

    def batch_search(self, release_id: str, dataset: str, id_type: str, 
                    search_ids: List[str]) -> Dict[str, Optional[IndexEntry]]:
        """
        Binary search for multiple IDs in the index.
        Returns a dictionary mapping search_id to IndexEntry (or None if not found).
        """
        try:
            results = {}
            # Sort search_ids to optimize memory access patterns
            search_ids = sorted(search_ids)
            
            for search_id in search_ids:
                results[search_id] = self.search(release_id, dataset, id_type, search_id)
                
            return results
            
        except Exception as e:
            console.print(f"[red]Error in batch search: {str(e)}[/red]")
            return {id: None for id in search_ids}

    def read_entry_data(self, entry: IndexEntry) -> Optional[dict]:
        """
        Read the JSON data for a given index entry.
        Returns None if the entry cannot be read.
        """
        try:
            file_path = Path(entry.file_path)
            if not file_path.exists():
                console.print(f"[red]File not found: {file_path}[/red]")
                return None
            
            with open(file_path, 'rb') as f:
                f.seek(entry.offset)
                line = f.readline()
                try:
                    # Try hex-encoded JSON first
                    decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                    return json.loads(decoded)
                except:
                    # Fall back to regular JSON
                    try:
                        return json.loads(line.strip())
                    except:
                        console.print(f"[red]Failed to parse JSON data at offset {entry.offset}[/red]")
                        return None
            
        except Exception as e:
            console.print(f"[red]Error reading entry data: {str(e)}[/red]")
            return None

    def get_index_stats(self, release_id: str) -> Dict[str, Dict]:
        """Get statistics about all indices for a release"""
        try:
            self._load_metadata(release_id)
            stats = {}
            
            for index_key, meta in self.metadata[release_id].items():
                try:
                    # Split on the first underscore
                    dataset, id_type = index_key.split('_', 1)
                    index_path = self._get_index_path(release_id, dataset, id_type)
                    
                    if not index_path.exists():
                        continue
                    
                    stats[index_key] = {
                        'entry_count': meta['entry_count'],
                        'size_mb': index_path.stat().st_size / (1024 * 1024),
                        'created': meta['created'],
                        'healthy': self._quick_health_check(index_path, meta)
                    }
                    
                except ValueError:
                    console.print(f"[yellow]Warning: Skipping malformed index key: {index_key}[/yellow]")
                    continue
                
            return stats
            
        except Exception as e:
            console.print(f"[red]Error getting index stats: {str(e)}[/red]")
            return {}
            
    def _quick_health_check(self, index_path: Path, meta: Dict) -> bool:
        """Perform a quick health check on an index file"""
        try:
            # Check file size
            if index_path.stat().st_size != meta['entry_count'] * IndexEntry.ENTRY_SIZE:
                return False
                
            # Read first and last entry to verify format
            with open(index_path, 'rb') as f:
                # Read first entry
                data = f.read(IndexEntry.ENTRY_SIZE)
                IndexEntry.from_bytes(data)
                
                # Read last entry
                f.seek(-IndexEntry.ENTRY_SIZE, 2)  # Seek from end
                data = f.read(IndexEntry.ENTRY_SIZE)
                IndexEntry.from_bytes(data)
                
            return True
            
        except Exception:
            return False

    def _count_entries_in_file(self, file_path: Path, sample_size: Optional[int] = None) -> Tuple[int, float]:
        """
        Count entries in a JSONL file, optionally using sampling for large files.
        Returns (total_count, confidence) where confidence is 1.0 for full counts
        and lower for sampled estimates.
        """
        try:
            file_size = file_path.stat().st_size
            
            # For small files (< 100MB), just do a full count
            if file_size < 100 * 1024 * 1024 or sample_size is None:
                with open(file_path, 'rb') as f:
                    count = 0
                    for line in f:
                        try:
                            # Try to decode hex-encoded JSON
                            decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                            json.loads(decoded)  # Validate it's valid JSON
                            count += 1
                        except:
                            # If hex decoding fails, try normal JSON
                            try:
                                json.loads(line.strip())
                                count += 1
                            except:
                                continue
                return count, 1.0
                
            # For large files, use sampling
            with open(file_path, 'rb') as f:
                # Read sample_size random positions
                positions = sorted(random.sample(range(file_size), sample_size))
                line_count = 0
                valid_samples = 0
                
                for pos in positions:
                    f.seek(pos)
                    # Skip partial line
                    f.readline()
                    # Read next full line
                    line = f.readline()
                    if line:
                        line_count += 1
                        try:
                            # Try hex decoding first
                            decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                            json.loads(decoded)
                            valid_samples += 1
                        except:
                            # If hex fails, try normal JSON
                            try:
                                json.loads(line.strip())
                                valid_samples += 1
                            except:
                                continue
                            
                # Estimate total lines based on sampling
                bytes_per_line = file_size / line_count if line_count else 0
                estimated_total = int((file_size / bytes_per_line) * (valid_samples / line_count)) if line_count else 0
                confidence = min(1.0, sample_size / estimated_total) if estimated_total > 0 else 0.0
                
                return estimated_total, confidence
                
        except Exception as e:
            console.print(f"[yellow]Warning: Error counting entries in {file_path}: {str(e)}[/yellow]")
            return 0, 0.0

    def verify_index_completeness(self, release_id: str, dataset: Optional[str] = None, quick_estimate: bool = False) -> bool:
        """
        Verify that indices contain all entries from source files by counting lines.
        Prints running totals as it processes files.
        
        Args:
            release_id: The release ID to verify
            dataset: Optional specific dataset to verify
            quick_estimate: If True, estimates total by sampling first file only
        """
        try:
            # Get all relevant datasets
            datasets_to_check = [dataset] if dataset else {
                k.split('_')[0] for k in self.metadata[release_id].keys()
            }

            total_lines = 0
            all_valid = True

            for dataset_name in datasets_to_check:
                dataset_dir = Path(self.base_dir) / release_id / dataset_name
                if not dataset_dir.exists():
                    console.print(f"[yellow]Dataset directory not found: {dataset_dir}[/yellow]")
                    continue

                # Get all JSON files except metadata
                files = [f for f in dataset_dir.glob("*.json") if f.name != 'metadata.json']
                if not files:
                    console.print(f"[yellow]No source files found for {dataset_name}[/yellow]")
                    continue

                dataset_lines = 0
                
                if quick_estimate:
                    # Just count lines in first file and multiply
                    first_file = files[0]
                    file_lines = 0
                    with open(first_file, 'rb') as f:
                        for line in f:
                            try:
                                # Try hex-encoded JSON first
                                bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                                file_lines += 1
                            except:
                                # Fall back to regular JSON
                                try:
                                    json.loads(line.strip())
                                    file_lines += 1
                                except:
                                    continue
                
                    estimated_total = file_lines * len(files)
                    console.print(
                        f"\n[bold]{dataset_name}[/bold]: Estimating ~{estimated_total:,} total lines "
                        f"(based on {file_lines:,} lines in {first_file.name} × {len(files)} files)"
                    )
                    dataset_lines = estimated_total
                    total_lines += estimated_total
                    
                else:
                    console.print(f"\n[bold]Counting lines in {dataset_name} files...[/bold]")
                    for file_path in files:
                        file_lines = 0
                        with open(file_path, 'rb') as f:
                            for line in f:
                                try:
                                    # Try hex-encoded JSON first
                                    bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                                    file_lines += 1
                                except:
                                    # Fall back to regular JSON
                                    try:
                                        json.loads(line.strip())
                                        file_lines += 1
                                    except:
                                        continue

                        dataset_lines += file_lines
                        total_lines += file_lines
                        console.print(f"{file_path.name}: {file_lines:,} lines (Running total: {total_lines:,})")

                # Compare with index counts for this dataset
                indices = {
                    k.split('_', 1)[1]: v 
                    for k, v in self.metadata[release_id].items() 
                    if k.startswith(f"{dataset_name}_")
                }

                if not indices:
                    console.print(f"[yellow]No indices found for {dataset_name}[/yellow]")
                    all_valid = False
                    continue

                for id_type, meta in indices.items():
                    index_count = meta['entry_count']
                    # Allow for 10% margin of error when using quick estimate
                    margin = int(dataset_lines * 0.1) if quick_estimate else 0
                    if abs(index_count - dataset_lines) > margin:
                        console.print(
                            f"[red]Count mismatch for {dataset_name}_{id_type}: "
                            f"Index has {index_count:,} entries, "
                            f"{'estimated' if quick_estimate else 'found'} {dataset_lines:,} lines in files "
                            f"({'±10%' if quick_estimate else 'exact'} comparison)[/red]"
                        )
                        all_valid = False
                    else:
                        console.print(
                            f"[green]✓ {dataset_name}_{id_type} index matches: {index_count:,} entries "
                            f"({'±10%' if quick_estimate else 'exact'} comparison)[/green]"
                        )

            console.print(
                f"\n[bold]Total {'estimated ' if quick_estimate else ''}lines across all files: {total_lines:,}[/bold]"
            )
            return all_valid

        except Exception as e:
            console.print(f"[red]Error verifying index completeness: {str(e)}[/red]")
            return False

    def lookup(self, release_id: str, dataset: str, id_type: str, search_id: str) -> Optional[Dict]:
        """
        Look up a record by ID in the binary index and return the parsed JSON.
        
        Args:
            release_id: The release ID (e.g. "2023-12-01")
            dataset: Dataset name (e.g. "papers", "abstracts", "s2orc")
            id_type: Type of ID to search for (e.g. "paper_id", "corpus_id")
            search_id: The ID to find
        
        Returns:
            Dict containing the parsed JSON record, or None if not found
        """
        try:
            # Search the binary index
            entry = self.search(release_id, dataset, id_type, search_id)
            if not entry:
                return None
            
            # Read and parse the JSON at the given offset
            with open(entry.file_path, 'r', encoding='utf-8') as f:
                f.seek(entry.offset)
                line = f.readline()
                try:
                    # Try hex-encoded JSON first
                    decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                    return json.loads(decoded)
                except:
                    # Fall back to regular JSON
                    try:
                        return json.loads(line.strip())
                    except:
                        console.print(f"[red]Failed to parse JSON at offset {entry.offset}[/red]")
                        return None
                    
        except Exception as e:
            console.print(f"[red]Error in binary index lookup: {str(e)}[/red]")
            return None