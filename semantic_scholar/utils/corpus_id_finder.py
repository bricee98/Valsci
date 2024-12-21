from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional, Tuple
from semantic_scholar.utils.binary_indexer import BinaryIndexer

console = Console()

class CorpusIDFinder:
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Use project root for base directory
            project_root = Path(__file__).parent.parent.parent
            base_dir = project_root / "semantic_scholar/datasets"
        
        self.base_dir = Path(base_dir)
        self.indexer = BinaryIndexer(self.base_dir)
        
        # Datasets that should contain corpus IDs
        self.corpus_id_datasets = [
            'papers',
            'abstracts',
            's2orc',
            'tldrs'
        ]

    def find_corpus_id(self, corpus_id: str, release_id: str = 'latest', show_index_samples: bool = True) -> Dict[str, Dict]:
        """
        Search for a corpus ID across all relevant datasets and their indices.
        Returns a dictionary with results from each dataset.
        """
        if release_id == 'latest':
            releases = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])
            if not releases:
                console.print("[red]No releases found in datasets directory[/red]")
                return {}
            release_id = releases[-1]

        results = {}
        
        # First check binary indices
        console.print(f"\n[cyan]Checking binary indices for release {release_id}...[/cyan]")
        for dataset in self.corpus_id_datasets:
            try:
                record = self.indexer.lookup(
                    release_id=release_id,
                    dataset=dataset,
                    id_type='corpus_id',
                    search_id=str(corpus_id)
                )
                
                if record:
                    results[dataset] = {
                        'found_in_index': True,
                        'record': record,
                        'file_info': None
                    }
                else:
                    results[dataset] = {
                        'found_in_index': False,
                        'record': None,
                        'file_info': None
                    }
            except Exception as e:
                results[dataset] = {
                    'found_in_index': False,
                    'record': None,
                    'error': str(e)
                }

        # Then check raw files
        console.print("\n[cyan]Checking raw dataset files...[/cyan]")
        for dataset in self.corpus_id_datasets:
            dataset_dir = self.base_dir / release_id / dataset
            if not dataset_dir.exists():
                continue

            found_info = self._search_files_for_corpus_id(dataset_dir, corpus_id)
            if found_info:
                if dataset not in results:
                    results[dataset] = {'found_in_index': False}
                results[dataset]['file_info'] = found_info

        # Add index inspection when ID not found
        if show_index_samples and not any(info.get('found_in_index') for info in results.values()):
            console.print("\n[yellow]ID not found in any index. Showing index samples for inspection...[/yellow]")
            for dataset in self.corpus_id_datasets:
                console.print(f"\n[bold]Inspecting {dataset} index:[/bold]")
                self.inspect_index(dataset, release_id, sample_size=3)

        # Display results in a table
        self._display_results(corpus_id, release_id, results)
        
        return results

    def _search_files_for_corpus_id(self, dataset_dir: Path, corpus_id: str) -> Optional[Dict]:
        """Search through raw files for a corpus ID."""
        for file_path in dataset_dir.glob("*.json"):
            if file_path.name == 'metadata.json':
                continue
                
            try:
                with open(file_path, 'r') as f:
                    line_number = 0
                    for line in f:
                        line_number += 1
                        try:
                            # Try hex-encoded JSON first
                            try:
                                decoded = bytes.fromhex(line.strip().decode('ascii')).decode('utf-8')
                                data = json.loads(decoded)
                            except:
                                # Fall back to regular JSON
                                data = json.loads(line.strip())
                            
                            # Check for corpus ID in various possible field names
                            found_id = data.get('corpusid') or data.get('corpus_id') or data.get('corpusId')
                            if str(found_id) == str(corpus_id):
                                return {
                                    'file': file_path.name,
                                    'line_number': line_number,
                                    'offset': f.tell()
                                }
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                console.print(f"[yellow]Error reading {file_path}: {str(e)}[/yellow]")
                continue
                
        return None

    def _display_results(self, corpus_id: str, release_id: str, results: Dict):
        """Display search results in a formatted table."""
        table = Table(title=f"Search Results for Corpus ID: {corpus_id}")
        
        table.add_column("Dataset")
        table.add_column("In Index")
        table.add_column("In Files")
        table.add_column("Details")
        
        for dataset, info in results.items():
            # Determine index status
            index_status = "[green]✓[/green]" if info.get('found_in_index') else "[red]✗[/red]"
            
            # Determine file status
            file_info = info.get('file_info')
            if file_info:
                file_status = f"[green]✓[/green]"
                details = f"File: {file_info['file']}\nLine: {file_info['line_number']}"
            else:
                file_status = "[red]✗[/red]"
                details = ""
            
            # Add error information if any
            if 'error' in info:
                details += f"\n[red]Error: {info['error']}[/red]"
            
            table.add_row(dataset, index_status, file_status, details)
        
        console.print(f"\n[bold]Results for release {release_id}:[/bold]")
        console.print(table)

    def inspect_index(self, dataset: str, release_id: str = 'latest', sample_size: int = 5) -> None:
        """
        Inspect a binary index file by showing sample entries.
        
        Args:
            dataset: Name of dataset to inspect
            release_id: Release ID to inspect
            sample_size: Number of sample entries to show
        """
        if release_id == 'latest':
            releases = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])
            if not releases:
                console.print("[red]No releases found in datasets directory[/red]")
                return
            release_id = releases[-1]

        index_path = self.indexer._get_index_path(release_id, dataset, 'corpus_id')
        if not index_path.exists():
            console.print(f"[red]No index file found at {index_path}[/red]")
            return

        try:
            # Get index metadata
            if release_id in self.indexer.metadata:
                meta = self.indexer.metadata[release_id].get(f"{dataset}_corpus_id", {})
                total_entries = meta.get('entry_count', 0)
            else:
                total_entries = index_path.stat().st_size // self.indexer.IndexEntry.ENTRY_SIZE

            console.print(f"\n[bold]Index File: {index_path.name}[/bold]")
            console.print(f"Total entries: {total_entries:,}")
            console.print(f"File size: {index_path.stat().st_size:,} bytes")

            # Show first few entries
            console.print(f"\n[cyan]First {sample_size} entries:[/cyan]")
            with open(index_path, 'rb') as f:
                for i in range(sample_size):
                    data = f.read(self.indexer.IndexEntry.ENTRY_SIZE)
                    if not data:
                        break
                    entry = self.indexer.IndexEntry.from_bytes(data)
                    console.print(f"{i+1}. ID: {entry.id}, File: {Path(entry.file_path).name}, Offset: {entry.offset}")

            # Show entries from middle
            mid_point = (total_entries // 2) - (sample_size // 2)
            console.print(f"\n[cyan]Middle {sample_size} entries (around position {mid_point:,}):[/cyan]")
            with open(index_path, 'rb') as f:
                f.seek(mid_point * self.indexer.IndexEntry.ENTRY_SIZE)
                for i in range(sample_size):
                    data = f.read(self.indexer.IndexEntry.ENTRY_SIZE)
                    if not data:
                        break
                    entry = self.indexer.IndexEntry.from_bytes(data)
                    console.print(f"{i+1}. ID: {entry.id}, File: {Path(entry.file_path).name}, Offset: {entry.offset}")

            # Show last few entries
            console.print(f"\n[cyan]Last {sample_size} entries:[/cyan]")
            with open(index_path, 'rb') as f:
                f.seek(-self.indexer.IndexEntry.ENTRY_SIZE * sample_size, 2)  # Seek from end
                for i in range(sample_size):
                    data = f.read(self.indexer.IndexEntry.ENTRY_SIZE)
                    if not data:
                        break
                    entry = self.indexer.IndexEntry.from_bytes(data)
                    console.print(f"{i+1}. ID: {entry.id}, File: {Path(entry.file_path).name}, Offset: {entry.offset}")

        except Exception as e:
            console.print(f"[red]Error inspecting index: {str(e)}[/red]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search for corpus IDs across datasets')
    parser.add_argument('corpus_id', help='Corpus ID to search for')
    parser.add_argument('--release', default='latest', help='Release ID to search in')
    parser.add_argument('--inspect', action='store_true', help='Show index samples even if ID is found')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample entries to show when inspecting')
    args = parser.parse_args()
    
    finder = CorpusIDFinder()
    if args.inspect:
        for dataset in finder.corpus_id_datasets:
            finder.inspect_index(dataset, args.release, args.samples)
    else:
        finder.find_corpus_id(args.corpus_id, args.release)

if __name__ == "__main__":
    main() 