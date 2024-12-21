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

    def find_corpus_id(self, corpus_id: str, release_id: str = 'latest') -> Dict[str, Dict]:
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search for corpus IDs across datasets')
    parser.add_argument('corpus_id', help='Corpus ID to search for')
    parser.add_argument('--release', default='latest', help='Release ID to search in')
    args = parser.parse_args()
    
    finder = CorpusIDFinder()
    finder.find_corpus_id(args.corpus_id, args.release)

if __name__ == "__main__":
    main() 