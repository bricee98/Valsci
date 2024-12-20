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

console = Console()

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
        return self.index_dir / f"{release_id}_{dataset}_{id_type}.idx"
        
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
            # Sort entries by ID for binary search
            entries.sort(key=lambda x: x.id)
            
            # Create unique temporary file
            tmp_path = self.tmp_dir / f"{release_id}_{dataset}_{id_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.idx.tmp"
            final_path = self._get_index_path(release_id, dataset, id_type)

            # Write sorted entries to temporary file
            with open(tmp_path, 'wb') as f:
                for entry in entries:
                    f.write(entry.to_bytes())

            # Verify the temporary index
            if verify:
                if not self._verify_index(tmp_path, entries):
                    raise ValueError("Index verification failed")

            # Calculate checksum
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
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(tmp_path), str(final_path))
            
            # Save updated metadata
            self._save_metadata(release_id)
            
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

    def verify_all_indices(self, release_id: str) -> bool:
        """Verify all indices for a release"""
        try:
            self._load_metadata(release_id)
            if release_id not in self.metadata:
                console.print(f"[yellow]No metadata found for release {release_id}[/yellow]")
                return False

            all_valid = True
            for index_key, meta in self.metadata[release_id].items():
                try:
                    # Split on the first underscore
                    dataset, id_type = index_key.split('_', 1)
                    index_path = self._get_index_path(release_id, dataset, id_type)
                    
                    # Verify file exists
                    if not index_path.exists():
                        console.print(f"[red]Index file missing: {index_path}[/red]")
                        all_valid = False
                        continue

                    # Verify file size
                    expected_size = meta['entry_count'] * IndexEntry.ENTRY_SIZE
                    actual_size = index_path.stat().st_size
                    if actual_size != expected_size:
                        console.print(f"[red]Size mismatch for {index_key}: expected {expected_size}, got {actual_size}[/red]")
                        all_valid = False
                        continue

                    # Verify checksum
                    current_checksum = self._calculate_file_checksum(index_path)
                    if current_checksum != meta['checksum']:
                        console.print(f"[red]Checksum mismatch for {index_key}[/red]")
                        all_valid = False
                        continue

                    console.print(f"[green]Verified {index_key}[/green]")

                except ValueError as e:
                    console.print(f"[red]Error processing index {index_key}: {str(e)}[/red]")
                    all_valid = False
                    continue

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
                
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(entry.offset)
                line = f.readline()
                return json.loads(line)
                
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