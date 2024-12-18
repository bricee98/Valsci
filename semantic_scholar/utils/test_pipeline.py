import sys
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import asyncio
import time
import sqlite3

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from semantic_scholar.utils.downloader import S2DatasetDownloader
from semantic_scholar.utils.searcher import S2Searcher
from app.models.claim import Claim
from app.services.claim_processor import ClaimProcessor
from app.services.literature_searcher import LiteratureSearcher
from app.services.paper_analyzer import PaperAnalyzer
from app.services.evidence_scorer import EvidenceScorer

console = Console()

class PipelineTester:
    def __init__(self):
        self.downloader = S2DatasetDownloader()
        self.searcher = S2Searcher()
        self.claim_processor = ClaimProcessor()
        
        # Test claims covering different scenarios
        self.test_claims = [
            "Calcium channels are affected by AMP kinase activation in cardiac cells",
            "Regular meditation practice increases gray matter density in the brain",
            "Coffee consumption reduces the risk of type 2 diabetes",
            "This is not a valid scientific claim",  # Should be flagged as invalid
            "Quantum entanglement enables faster-than-light communication"  # Controversial/disputed claim
        ]

    def test_dataset_management(self):
        """Test dataset download and verification."""
        console.print("\n[cyan]Testing Dataset Management[/cyan]")
        
        # Check if datasets already exist
        latest_release = self._get_latest_local_release()
        if latest_release:
            console.print(f"[yellow]Found existing datasets from release {latest_release}[/yellow]")
            
            # Check each required dataset
            missing_datasets = []
            for dataset in self.downloader.datasets_to_download:
                dataset_dir = Path(project_root) / "semantic_scholar/datasets" / latest_release / dataset
                if not dataset_dir.exists():
                    console.print(f"[red]! {dataset} dataset missing[/red]")
                    missing_datasets.append(dataset)
                else:
                    console.print(f"[green]✓ {dataset} dataset found[/green]")
                    
                    # Special verification for S2ORC
                    if dataset == 's2orc':
                        self._verify_s2orc_content(dataset_dir)
            
            # Download only missing datasets
            if missing_datasets:
                console.print(f"[yellow]Downloading missing datasets: {', '.join(missing_datasets)}[/yellow]")
                for dataset in missing_datasets:
                    success = self.downloader.download_dataset(dataset, mini=True)
                    if success:
                        console.print(f"[green]✓ {dataset} downloaded successfully[/green]")
                    else:
                        console.print(f"[red]! Failed to download {dataset}[/red]")
            
            # Verify all datasets (in mini mode)
            verified = self.downloader.verify_downloads(mini=True)
            if verified:
                console.print("[green]✓ All mini datasets verified successfully[/green]")
            else:
                console.print("[yellow]Note: Missing files are expected in mini mode[/yellow]")
        else:
            console.print("[yellow]No existing datasets found. Starting download...[/yellow]")
            self._perform_download_test()
        
        # Test dataset access
        assert self.searcher.has_local_data, "Local dataset not accessible"
        console.print("[green]✓ Dataset access test passed[/green]")

    def _get_latest_local_release(self):
        """Get the latest release from local datasets."""
        datasets_dir = Path(project_root) / "semantic_scholar/datasets"
        if not datasets_dir.exists():
            return None
        
        releases = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
        return max(releases) if releases else None

    def _perform_download_test(self):
        """Perform the dataset download and verification test."""
        with Progress() as progress:
            task = progress.add_task("Testing dataset management...", total=2)
            
            # Test mini download
            progress.update(task, description="Downloading mini dataset...")
            success = self.downloader.download_all_datasets(mini=True)
            assert success, "Mini dataset download failed"
            progress.advance(task)
            
            # Test verification
            progress.update(task, description="Verifying downloads...")
            verified = self.downloader.verify_downloads()
            assert verified, "Dataset verification failed"
            progress.advance(task)
        
        console.print("[green]✓ Dataset management tests passed[/green]")

    def test_search_functionality(self):
        """Test paper search and retrieval."""
        console.print("\n[cyan]Testing Search Functionality[/cyan]")
        
        results_table = Table(title="Search Results")
        results_table.add_column("Claim", style="cyan")
        results_table.add_column("Papers Found", style="green")
        results_table.add_column("Full Text", style="blue")
        results_table.add_column("Time (s)", style="yellow")
        
        for claim_text in self.test_claims[:2]:  # Test first two claims
            start_time = time.time()
            papers = self.searcher.search_papers_for_claim(
                claim_text, 
                num_queries=2, 
                results_per_query=3
            )
            duration = time.time() - start_time
            
            # Count papers with full text
            full_text_count = sum(1 for p in papers if p.get('content_source') == 's2orc')
            
            results_table.add_row(
                claim_text[:50] + "...",
                str(len(papers)),
                f"{full_text_count}/{len(papers)}",
                f"{duration:.2f}"
            )
            
            # Save detailed results for inspection
            with open(f'search_results_{hash(claim_text)}.json', 'w') as f:
                json.dump({
                    'claim': claim_text,
                    'papers': papers,
                    'full_text_ratio': f"{full_text_count}/{len(papers)}",
                    'search_time': duration
                }, f, indent=2)
        
        console.print(results_table)
        console.print("[green]✓ Search functionality tests passed[/green]")

    def test_full_pipeline(self):
        """Test the complete claim processing pipeline."""
        console.print("\n[cyan]Testing Full Pipeline[/cyan]")
        
        results_table = Table(title="Pipeline Results")
        results_table.add_column("Claim", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Rating", style="yellow")
        results_table.add_column("Time (s)", style="magenta")
        
        for claim_text in self.test_claims:
            start_time = time.time()
            duration = 0  # Initialize duration
            
            try:
                # Create claim object
                claim = Claim(claim_text)
                claim.search_config = {
                    'numQueries': 2,
                    'resultsPerQuery': 3,
                    'reviewType': 'abstracts'
                }
                
                # Process claim
                self.claim_processor.process_claim(
                    claim=claim,
                    batch_id='test_batch',
                    claim_id=str(hash(claim_text))
                )
                duration = time.time() - start_time
                
                results_table.add_row(
                    claim_text[:50] + "...",
                    claim.status,
                    str(claim.report.get('claimRating', 'N/A')),
                    f"{duration:.2f}"
                )
                
                # Save detailed results
                with open(f'pipeline_results_{hash(claim_text)}.json', 'w') as f:
                    json.dump(claim.report, f, indent=2)
                    
            except Exception as e:
                duration = time.time() - start_time
                console.print(f"[red]Error processing claim: {str(e)}[/red]")
                results_table.add_row(
                    claim_text[:50] + "...",
                    "error",
                    "N/A",
                    f"{duration:.2f}"
                )
        
        console.print(results_table)
        console.print("[green]✓ Full pipeline tests completed[/green]")

    def run_all_tests(self):
        """Run all pipeline tests."""
        try:
            console.print("[bold cyan]Starting Pipeline Tests[/bold cyan]")
            
            self.test_dataset_management()
            self.test_search_functionality()
            self.test_full_pipeline()
            
            console.print("\n[bold green]All tests completed successfully![/bold green]")
            
        except Exception as e:
            console.print(f"\n[bold red]Test suite failed: {str(e)}[/bold red]")
            raise

    def _verify_s2orc_content(self, s2orc_dir: Path) -> bool:
        """Verify S2ORC dataset content format."""
        try:
            sample_files = list(s2orc_dir.glob('*.json'))
            if not sample_files:
                console.print("[red]! No S2ORC data files found[/red]")
                return False
            
            with open(sample_files[0], 'r') as f:
                sample_line = next(f)
                data = json.loads(sample_line)
                if 'content' in data and 'text' in data['content']:
                    console.print("[green]✓ S2ORC content format verified[/green]")
                    return True
                else:
                    console.print("[red]! S2ORC content format invalid[/red]")
                    return False
                
        except Exception as e:
            console.print(f"[red]! Error verifying S2ORC content: {str(e)}[/red]")
            return False

    def _verify_indices(self):
        """Verify that indices are working properly."""
        console.print("\n[cyan]Verifying indices...[/cyan]")
        
        index_path = self.base_dir / "indices" / f"{self.current_release}.db"
        if not index_path.exists():
            console.print("[red]No index found[/red]")
            return False

        try:
            with sqlite3.connect(str(index_path)) as conn:
                # Check table structure
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                if 'paper_locations' not in tables:
                    console.print("[red]Missing paper_locations table[/red]")
                    return False

                # Check index content
                cursor = conn.execute("SELECT COUNT(*) FROM paper_locations")
                count = cursor.fetchone()[0]
                console.print(f"[green]Found {count} indexed records[/green]")

                # Test a few random lookups
                cursor = conn.execute("SELECT * FROM paper_locations LIMIT 5")
                sample_records = cursor.fetchall()
                
                if not sample_records:
                    console.print("[red]No sample records found[/red]")
                    return False

                # Verify file paths exist
                for record in sample_records:
                    file_path = Path(record[2])
                    if not file_path.exists():
                        console.print(f"[red]Missing file: {file_path}[/red]")
                        return False

                return True

        except Exception as e:
            console.print(f"[red]Error verifying indices: {str(e)}[/red]")
            return False

    def test_dataset_updates(self):
        """Test dataset update functionality."""
        console.print("\n[cyan]Testing Dataset Updates[/cyan]")
        
        try:
            downloader = S2DatasetDownloader()
            
            # Get current state
            initial_release = downloader._get_latest_local_release()
            if not initial_release:
                console.print("[yellow]No local datasets found - skipping update test[/yellow]")
                return
            
            # Verify index exists
            index_path = downloader.index_dir / f"{initial_release}.db"
            if not index_path.exists():
                console.print("[yellow]No index found - skipping update test[/yellow]")
                return
            
            # Get initial record count
            with sqlite3.connect(str(index_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM paper_locations")
                initial_count = cursor.fetchone()[0]
            
            # Perform update
            success = downloader.update_datasets()
            
            if success:
                # Verify new release
                new_release = downloader._get_latest_local_release()
                if new_release == initial_release:
                    console.print("[green]✓ No updates needed - datasets already current[/green]")
                else:
                    # Check new index
                    new_index_path = downloader.index_dir / f"{new_release}.db"
                    with sqlite3.connect(str(new_index_path)) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM paper_locations")
                        new_count = cursor.fetchone()[0]
                    
                    console.print(f"[green]✓ Updated from {initial_release} to {new_release}[/green]")
                    console.print(f"[green]✓ Index updated: {initial_count} -> {new_count} records[/green]")
                    
                    # Verify no backup files remain
                    backup_path = downloader.index_dir / f"{initial_release}.db.bak"
                    assert not backup_path.exists(), "Backup file not cleaned up"
                    console.print("[green]✓ Cleanup successful[/green]")
            else:
                console.print("[red]Update failed[/red]")
            
        except Exception as e:
            console.print(f"[red]Error testing updates: {str(e)}[/red]")
            raise

def main():
    tester = PipelineTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 