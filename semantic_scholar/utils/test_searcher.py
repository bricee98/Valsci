import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
import json
import ijson
import time
import sqlite3

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from semantic_scholar.utils.searcher import S2Searcher

console = Console()

def validate_dataset_structure():
    """Validate the structure of downloaded datasets."""
    base_dir = Path(project_root) / "semantic_scholar/datasets"
    if not base_dir.exists():
        console.print("[red]Error: No datasets directory found[/red]")
        return False

    releases = [d for d in base_dir.iterdir() if d.is_dir()]
    if not releases:
        console.print("[red]Error: No releases found in datasets directory[/red]")
        return False

    latest_release = max(releases)
    console.print(f"\n[cyan]Validating datasets in release: {latest_release.name}[/cyan]")

    expected_datasets = {
        'papers': {
            'required_fields': {'corpusid'},
            'sample_count': 0
        },
        'abstracts': {
            'required_fields': {'corpusid'},
            'sample_count': 0
        },
        'citations': {
            'required_fields': {'citingcorpusid'},
            'sample_count': 0
        },
        'authors': {
            'required_fields': {'authorid'},
            'sample_count': 0
        }
    }

    validation_table = Table(title="Dataset Validation Results")
    validation_table.add_column("Dataset", style="cyan")
    validation_table.add_column("Status", style="green")
    validation_table.add_column("Sample Count", style="blue")
    validation_table.add_column("Issues", style="red")

    all_valid = True
    for dataset_name, requirements in expected_datasets.items():
        dataset_dir = latest_release / dataset_name
        if not dataset_dir.exists():
            validation_table.add_row(
                dataset_name,
                "❌ Failed",
                "0",
                "Directory not found"
            )
            all_valid = False
            continue

        json_files = list(dataset_dir.glob('*.json'))
        if not json_files:
            validation_table.add_row(
                dataset_name,
                "❌ Failed",
                "0",
                "No JSON files found"
            )
            all_valid = False
            continue

        # Check first data file (excluding metadata.json)
        data_files = [f for f in json_files if f.name != 'metadata.json']
        if not data_files:
            validation_table.add_row(
                dataset_name,
                "❌ Failed",
                "0",
                "No data files found"
            )
            all_valid = False
            continue

        sample_file = data_files[0]
        issues = []
        try:
            # Count items and validate structure (JSONL format)
            item_count = 0
            required_fields = requirements['required_fields']
            
            with open(sample_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # Check if all required fields are present
                        if not required_fields.issubset(item.keys()):
                            missing = required_fields - set(item.keys())
                            issues.append(f"Missing fields: {missing}")
                            break
                        item_count += 1
                        if item_count >= 5:  # Check first 5 items
                            break
                    except json.JSONDecodeError as e:
                        issues.append(f"Invalid JSON in line: {str(e)}")
                        break

            expected_datasets[dataset_name]['sample_count'] = item_count
            
            if issues:
                validation_table.add_row(
                    dataset_name,
                    "❌ Failed",
                    str(item_count),
                    ", ".join(issues)
                )
                all_valid = False
            else:
                validation_table.add_row(
                    dataset_name,
                    "✓ Valid",
                    str(item_count),
                    ""
                )

        except Exception as e:
            validation_table.add_row(
                dataset_name,
                "❌ Failed",
                "0",
                f"Error reading file: {str(e)}"
            )
            all_valid = False

    console.print(validation_table)
    return all_valid

def test_search_functionality():
    """Test basic search functionality."""
    # First validate datasets
    console.print("\n[cyan]Validating downloaded datasets...[/cyan]")
    datasets_valid = validate_dataset_structure()
    if not datasets_valid:
        console.print("[yellow]Warning: Dataset validation failed, continuing in API-only mode[/yellow]")

    searcher = S2Searcher()
    
    # Test claim
    claim = "Calcium channels are affected by AMP kinase activation in cardiac cells"
    
    console.print("\n[cyan]Testing search query generation...[/cyan]")
    queries = searcher.generate_search_queries(claim, num_queries=3)
    
    table = Table(title="Generated Search Queries")
    table.add_column("Query", style="green")
    for query in queries:
        table.add_row(query)
    console.print(table)

    console.print("\n[cyan]Testing paper search...[/cyan]")
    papers = searcher.search_papers_for_claim(claim, num_queries=2, results_per_query=3)
    
    results_table = Table(title="Search Results")
    results_table.add_column("Title", style="green")
    results_table.add_column("Year", style="blue")
    results_table.add_column("Citations", style="yellow")
    results_table.add_column("Authors", style="magenta")
    
    for paper in papers:
        authors = ", ".join([
            f"{a['name']} (h={a.get('hIndex', 'N/A')})" 
            for a in paper.get('authors', [])[:2]
        ])
        if len(paper.get('authors', [])) > 2:
            authors += " et al."
            
        results_table.add_row(
            paper.get('title', 'N/A'),
            str(paper.get('year', 'N/A')),
            str(paper.get('citation_count', 'N/A')),
            authors
        )
    
    console.print(results_table)

    # Save detailed results for inspection
    console.print("\n[cyan]Saving detailed results to 'test_results.json'...[/cyan]")
    with open('test_results.json', 'w') as f:
        json.dump(papers, f, indent=2)

def test_index_functionality():
    """Test that indices are working properly."""
    searcher = S2Searcher()
    
    # First verify index exists
    index_path = searcher.base_dir / "indices" / f"{searcher.current_release}.db"
    if not index_path.exists():
        console.print("[red]No index found. Run downloader first.[/red]")
        return False

    # Test queries using sample data from ex_sample_lines.json
    test_cases = [
        {
            'dataset': 'papers',
            'corpus_id': '71452834',  # From sample papers line
            'expected_fields': {'title', 'authors'}
        },
        {
            'dataset': 'abstracts',
            'corpus_id': '150777384',  # From sample abstracts line
            'expected_fields': {'abstract'}
        },
        {
            'dataset': 's2orc',
            'corpus_id': '16385537',  # From sample s2orc line
            'expected_fields': {'content'}
        }
    ]

    results_table = Table(title="Index Test Results")
    results_table.add_column("Dataset", style="cyan")
    results_table.add_column("Corpus ID", style="blue")
    results_table.add_column("Found", style="green")
    results_table.add_column("Fields Present", style="yellow")
    results_table.add_column("Lookup Time (ms)", style="magenta")

    all_passed = True
    for test in test_cases:
        start_time = time.time()
        result = searcher._find_in_dataset(test['dataset'], test['corpus_id'])
        lookup_time = (time.time() - start_time) * 1000  # Convert to ms

        if result:
            found_fields = set(result.keys())
            fields_present = all(field in found_fields for field in test['expected_fields'])
            status = "✓" if fields_present else "!"
            fields_str = ", ".join(found_fields & test['expected_fields'])
        else:
            status = "✗"
            fields_str = "N/A"
            all_passed = False

        results_table.add_row(
            test['dataset'],
            test['corpus_id'],
            status,
            fields_str,
            f"{lookup_time:.2f}"
        )

    console.print(results_table)

    # Test random access performance
    console.print("\n[cyan]Testing random access performance...[/cyan]")
    
    # Get a list of corpus IDs from the index
    with sqlite3.connect(str(index_path)) as conn:
        cursor = conn.execute("SELECT DISTINCT corpus_id FROM paper_locations LIMIT 100")
        corpus_ids = [row[0] for row in cursor.fetchall()]

    if corpus_ids:
        # Time lookups for random corpus IDs
        import random
        test_size = min(10, len(corpus_ids))
        total_time = 0
        
        perf_table = Table(title="Random Access Performance")
        perf_table.add_column("Corpus ID", style="blue")
        perf_table.add_column("Found", style="green")
        perf_table.add_column("Time (ms)", style="yellow")

        for _ in range(test_size):
            corpus_id = random.choice(corpus_ids)
            start_time = time.time()
            result = searcher._find_in_dataset('papers', corpus_id)
            lookup_time = (time.time() - start_time) * 1000
            total_time += lookup_time

            perf_table.add_row(
                corpus_id,
                "✓" if result else "✗",
                f"{lookup_time:.2f}"
            )

        console.print(perf_table)
        console.print(f"\nAverage lookup time: {total_time/test_size:.2f}ms")
    
    return all_passed

def main():
    try:
        # Run existing tests
        test_search_functionality()
        
        # Test index functionality
        console.print("\n[cyan]Testing index functionality...[/cyan]")
        if test_index_functionality():
            console.print("[green]✓ Index tests passed[/green]")
        else:
            console.print("[red]✗ Index tests failed[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    main() 