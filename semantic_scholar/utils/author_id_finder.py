from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional, Tuple
from semantic_scholar.utils.binary_indexer import BinaryIndexer, IndexEntry

console = Console()

class AuthorIDFinder:
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Use project root for base directory
            project_root = Path(__file__).parent.parent.parent
            base_dir = project_root / "semantic_scholar/datasets"
        
        self.base_dir = Path(base_dir)
        self.indexer = BinaryIndexer(self.base_dir)
        
        # Load metadata to get latest release
        self.current_release = self._get_latest_release()
        
        # Store IndexEntry class for convenience
        self.IndexEntry = IndexEntry
        
        # Datasets that should contain author IDs
        self.author_id_datasets = [
            'authors'  # Only search in authors dataset
        ]

    def _get_latest_release(self) -> Optional[str]:
        """Get the latest release ID from metadata files."""
        metadata_files = list(self.base_dir.glob("binary_indices/*_metadata.json"))
        if not metadata_files:
            return None
        # Extract release IDs from metadata filenames and get the latest
        releases = [f.name.split('_')[0] for f in metadata_files]
        return max(releases) if releases else None

    def find_author_id(self, author_id: str, release_id: str = 'latest', show_index_samples: bool = True) -> Dict[str, Dict]:
        """
        Search for an author ID across all relevant datasets and their indices.
        Returns a dictionary with results from each dataset.
        """
        if release_id == 'latest':
            release_id = self.current_release
            if not release_id:
                console.print("[red]No release metadata found[/red]")
                return {}

        results = {}
        
        # First check binary indices
        console.print(f"\n[cyan]Checking binary indices for release {release_id}...[/cyan]")
        for dataset in self.author_id_datasets:
            try:
                # First get the index entry
                index_entry = self.indexer.search(
                    release_id=release_id,
                    dataset=dataset,
                    id_type='author_id',
                    search_id=str(author_id)
                )
                
                # Then get the record data if we found an entry
                record = None
                if index_entry:
                    record = self.indexer.read_entry_data(index_entry)
                
                if index_entry:
                    results[dataset] = {
                        'found_in_index': True,
                        'index_entry': index_entry,  # Store the actual IndexEntry
                        'record': record,
                        'file_info': None
                    }
                else:
                    results[dataset] = {
                        'found_in_index': False,
                        'index_entry': None,
                        'record': None,
                        'file_info': None
                    }
            except Exception as e:
                results[dataset] = {
                    'found_in_index': False,
                    'index_entry': None,
                    'record': None,
                    'error': str(e)
                }

        # Then check raw files
        console.print("\n[cyan]Checking raw dataset files...[/cyan]")
        for dataset in self.author_id_datasets:
            dataset_dir = self.base_dir / release_id / dataset
            if not dataset_dir.exists():
                continue

            found_info = self._search_files_for_author_id(dataset_dir, author_id)
            if found_info:
                if dataset not in results:
                    results[dataset] = {'found_in_index': False}
                results[dataset]['file_info'] = found_info

        # Add index inspection when ID not found
        if show_index_samples and not any(info.get('found_in_index') for info in results.values()):
            console.print("\n[yellow]ID not found in any index. Showing index samples for inspection...[/yellow]")
            for dataset in self.author_id_datasets:
                console.print(f"\n[bold]Inspecting {dataset} index:[/bold]")
                self.inspect_index(dataset, release_id, sample_size=3)

        # Display results in a table
        self._display_results(author_id, release_id, results)
        
        return results

    def _search_files_for_author_id(self, dataset_dir: Path, author_id: str) -> Optional[Dict]:
        """Search for an author ID using the index to find the correct file."""
        try:
            # First get the index entry
            entry = self.indexer.search(
                release_id=self.current_release,
                dataset=dataset_dir.name,
                id_type='author_id',
                search_id=str(author_id)
            )
            
            if not entry:
                return None
                
            # Use the index entry to go directly to the right file and offset
            file_path = Path(entry.file_path)
            if not file_path.exists():
                console.print(f"[yellow]Warning: Index points to missing file: {file_path}[/yellow]")
                return None
                
            with open(file_path, 'rb') as f:  # Open in binary mode
                f.seek(entry.offset)
                line = f.readline()
                try:
                    # Try hex-encoded JSON first
                    try:
                        decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                        data = json.loads(decoded)
                    except:
                        # Fall back to regular JSON
                        data = json.loads(line.strip().decode('utf-8'))
                    
                    # Verify we found the right record
                    found_id = str(data.get('authorid') or data.get('authorId'))
                    if found_id == str(author_id):
                        return {
                            'file': file_path.name,
                            'offset': entry.offset,
                            'data': data
                        }
                    else:
                        console.print(f"[yellow]Warning: Index pointed to wrong record (found ID: {found_id})[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]Error parsing JSON at offset {entry.offset}: {str(e)}[/yellow]")
                    
            return None
                    
        except Exception as e:
            console.print(f"[yellow]Error searching files: {str(e)}[/yellow]")
            return None

    def _display_results(self, author_id: str, release_id: str, results: Dict):
        """Display search results in a formatted table."""
        table = Table(title=f"Search Results for Author ID: {author_id}")
        
        table.add_column("Dataset")
        table.add_column("In Index")
        table.add_column("In Files")
        table.add_column("Details")
        
        for dataset, info in results.items():
            # Determine index status
            index_status = "[green]✓[/green]" if info.get('found_in_index') else "[red]✗[/red]"
            
            # Determine file status and details
            file_info = info.get('file_info')
            details = []
            
            # Add index record info if found
            if info.get('index_entry'):  # Use index_entry instead of record
                entry = info['index_entry']
                details.append("[cyan]Index points to:[/cyan]")
                details.append(f"File: {Path(entry.file_path).name}")
                details.append(f"Offset: {entry.offset}")
            
            # Add file info if found
            if file_info:
                file_status = f"[green]✓[/green]"
                if file_info.get('file'):
                    details.append(f"Found in file: {file_info['file']}")
                if file_info.get('offset'):
                    details.append(f"At offset: {file_info['offset']}")
                if file_info.get('data'):
                    # Show a preview of the data
                    data_preview = str(file_info['data'])[:100] + "..." if len(str(file_info['data'])) > 100 else str(file_info['data'])
                    details.append(f"Data: {data_preview}")
            else:
                file_status = "[red]✗[/red]"
            
            # Add error information if any
            if 'error' in info:
                details.append(f"[red]Error: {info['error']}[/red]")
            
            table.add_row(dataset, index_status, file_status, "\n".join(details))
        
        console.print(f"\n[bold]Results for release {release_id}:[/bold]")
        console.print(table)

    def inspect_index(self, dataset: str, release_id: str = 'latest', sample_size: int = 5) -> None:
        """
        Inspect a binary index file by showing sample entries.
        """
        if release_id == 'latest':
            release_id = self.current_release
            if not release_id:
                console.print("[red]No release metadata found[/red]")
                return

        # Construct correct index path
        index_path = self.base_dir / "binary_indices" / f"{release_id}_{dataset}_author_id.idx"
        if not index_path.exists():
            console.print(f"[red]No index file found at {index_path}[/red]")
            return

        try:
            # Get index metadata
            metadata_path = self.base_dir / "binary_indices" / f"{release_id}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                meta = metadata.get(f"{dataset}_author_id", {})
                total_entries = meta.get('entry_count', 0)
            else:
                total_entries = index_path.stat().st_size // IndexEntry.ENTRY_SIZE

            console.print(f"\n[bold]Index File: {index_path.name}[/bold]")
            console.print(f"Total entries: {total_entries:,}")
            console.print(f"File size: {index_path.stat().st_size:,} bytes")

            # Show first few entries
            console.print(f"\n[cyan]First {sample_size} entries:[/cyan]")
            with open(index_path, 'rb') as f:
                for i in range(sample_size):
                    data = f.read(IndexEntry.ENTRY_SIZE)
                    if not data:
                        break
                    entry = IndexEntry.from_bytes(data)
                    console.print(f"{i+1}. ID: {entry.id}, File: {Path(entry.file_path).name}, Offset: {entry.offset}")

            # Show entries from middle
            mid_point = (total_entries // 2) - (sample_size // 2)
            console.print(f"\n[cyan]Middle {sample_size} entries (around position {mid_point:,}):[/cyan]")
            with open(index_path, 'rb') as f:
                f.seek(mid_point * IndexEntry.ENTRY_SIZE)
                for i in range(sample_size):
                    data = f.read(IndexEntry.ENTRY_SIZE)
                    if not data:
                        break
                    entry = IndexEntry.from_bytes(data)
                    console.print(f"{i+1}. ID: {entry.id}, File: {Path(entry.file_path).name}, Offset: {entry.offset}")

            # Show last few entries
            console.print(f"\n[cyan]Last {sample_size} entries:[/cyan]")
            with open(index_path, 'rb') as f:
                f.seek(-IndexEntry.ENTRY_SIZE * sample_size, 2)  # Seek from end
                for i in range(sample_size):
                    data = f.read(IndexEntry.ENTRY_SIZE)
                    if not data:
                        break
                    entry = IndexEntry.from_bytes(data)
                    console.print(f"{i+1}. ID: {entry.id}, File: {Path(entry.file_path).name}, Offset: {entry.offset}")

        except Exception as e:
            console.print(f"[red]Error inspecting index: {str(e)}[/red]")
            raise  # Add this to see full traceback during development

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search for author IDs across datasets')
    parser.add_argument('author_id', help='Author ID to search for')
    parser.add_argument('--release', default='latest', help='Release ID to search in')
    parser.add_argument('--inspect', action='store_true', help='Show index samples even if ID is found')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample entries to show when inspecting')
    args = parser.parse_args()
    
    finder = AuthorIDFinder()
    # Always do the search first
    results = finder.find_author_id(args.author_id, args.release)
    
    # Then show index samples if requested
    if args.inspect:
        for dataset in finder.author_id_datasets:
            finder.inspect_index(dataset, args.release, args.samples)

if __name__ == "__main__":
    main() 