import os
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console
from rich.table import Table
from .binary_indexer import BinaryIndexer

console = Console()

class PaperSearcher:
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the paper searcher.
        
        Args:
            base_dir: Optional base directory for datasets. If not provided,
                     uses the semantic_scholar/datasets directory relative to project root.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Default to semantic_scholar/datasets relative to project root
            self.base_dir = Path(__file__).parent.parent / "datasets"
        
        self.indexer = BinaryIndexer(self.base_dir)
        
    def search_paper(self, paper_id: str, release_id: str = 'latest') -> Dict:
        """
        Search for a paper and its related information across multiple datasets.
        
        Args:
            paper_id: The paper ID to search for (can be paper_id or corpus_id)
            release_id: Release ID to search in (defaults to latest)
            
        Returns:
            Dictionary containing all found information about the paper
        """
        try:
            # Determine if this is a paper_id or corpus_id
            is_corpus_id = paper_id.isdigit()
            id_type = 'corpus_id' if is_corpus_id else 'paper_id'
            
            results = {
                'paper': None,
                'abstract': None,
                's2orc': None,
                'tldr': None,
                'release_id': release_id
            }
            
            # If using latest, get the actual release ID
            if release_id == 'latest':
                releases = [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
                if not releases:
                    console.print("[red]No releases found in datasets directory[/red]")
                    return results
                release_id = sorted(releases)[-1]
                results['release_id'] = release_id
            
            # Search in papers dataset
            if is_corpus_id:
                paper_data = self.indexer.lookup(release_id, 'papers', 'corpus_id', paper_id)
            else:
                paper_data = self.indexer.lookup(release_id, 'papers', 'paper_id', paper_id)
            
            if paper_data:
                results['paper'] = paper_data
                # Get corpus_id for searching other datasets
                corpus_id = str(paper_data.get('corpusid'))
                
                # Search for abstract
                abstract_data = self.indexer.lookup(release_id, 'abstracts', 'corpus_id', corpus_id)
                if abstract_data:
                    results['abstract'] = abstract_data
                
                # Search for S2ORC full text
                s2orc_data = self.indexer.lookup(release_id, 's2orc', 'corpus_id', corpus_id)
                if s2orc_data:
                    results['s2orc'] = s2orc_data
                
                # Search for TLDR
                tldr_data = self.indexer.lookup(release_id, 'tldrs', 'corpus_id', corpus_id)
                if tldr_data:
                    results['tldr'] = tldr_data
            
            return results
            
        except Exception as e:
            console.print(f"[red]Error searching for paper: {str(e)}[/red]")
            return None
        
    def display_results(self, results: Dict):
        """Display search results in a formatted table."""
        if not results:
            console.print("[red]No results to display[/red]")
            return
            
        console.print(f"\n[bold cyan]Search Results (Release: {results['release_id']})[/bold cyan]")
        
        # Paper information
        if results['paper']:
            paper = results['paper']
            table = Table(title="Paper Information")
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            
            table.add_row("Title", paper.get('title', 'N/A'))
            table.add_row("Paper ID", paper.get('paperId', 'N/A'))
            table.add_row("Corpus ID", str(paper.get('corpusid', 'N/A')))
            table.add_row("Year", str(paper.get('year', 'N/A')))
            table.add_row("Venue", paper.get('venue', 'N/A'))
            
            authors = paper.get('authors', [])
            author_names = [a.get('name', 'Unknown') for a in authors]
            table.add_row("Authors", ", ".join(author_names))
            
            console.print(table)
            
            # Abstract
            if results['abstract']:
                console.print("\n[bold]Abstract:[/bold]")
                console.print(results['abstract'].get('abstract', 'N/A'))
            
            # TLDR
            if results['tldr']:
                console.print("\n[bold]TLDR:[/bold]")
                console.print(results['tldr'].get('tldr', 'N/A'))
            
            # S2ORC
            if results['s2orc']:
                s2orc = results['s2orc']
                console.print("\n[bold]S2ORC Full Text Available:[/bold]")
                console.print(f"• PDF Hash: {s2orc.get('pdfHash', 'N/A')}")
                console.print(f"• Total References: {len(s2orc.get('references', []))}")
                console.print(f"• Total Citations: {len(s2orc.get('citations', []))}")
        else:
            console.print("[yellow]No paper found with this ID[/yellow]")

def main():
    """Command line interface for paper searching."""
    import argparse
    parser = argparse.ArgumentParser(description='Search for papers across multiple datasets')
    parser.add_argument('paper_id', help='Paper ID or Corpus ID to search for')
    parser.add_argument('--release', default='latest', help='Release ID to search in')
    parser.add_argument('--raw', action='store_true', help='Display raw JSON results')
    args = parser.parse_args()
    
    searcher = PaperSearcher()
    results = searcher.search_paper(args.paper_id, args.release)
    
    if args.raw:
        import json
        print(json.dumps(results, indent=2))
    else:
        searcher.display_results(results)

if __name__ == "__main__":
    main() 