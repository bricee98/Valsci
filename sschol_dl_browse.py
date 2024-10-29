import requests
import json
import sys
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import math
from datetime import datetime
import os
from rich.text import Text

BASE_URL = "https://api.semanticscholar.org/datasets/v1"
console = Console()

class S2AGClient:
    def __init__(self):
        self.session = requests.Session()
    
    def get_releases(self) -> List[str]:
        """Get all available releases."""
        response = self.session.get(f"{BASE_URL}/release")
        response.raise_for_status()
        return response.json()
    
    def get_latest_release(self) -> str:
        """Get the latest release ID."""
        response = self.session.get(f"{BASE_URL}/release/latest")
        response.raise_for_status()
        return response.json()["release_id"]
    
    def get_release_info(self, release_id: str) -> Dict:
        """Get detailed information about a specific release."""
        response = self.session.get(f"{BASE_URL}/release/{release_id}")
        response.raise_for_status()
        return response.json()
    
    def get_dataset_info(self, release_id: str, dataset_name: str) -> Dict:
        """Get information about a specific dataset including download links."""
        response = self.session.get(f"{BASE_URL}/release/{release_id}/dataset/{dataset_name}")
        response.raise_for_status()
        return response.json()

    def estimate_dataset_size(self, url: str) -> int:
        """Estimate file size using HEAD request."""
        try:
            response = self.session.head(url)
            return int(response.headers.get('content-length', 0))
        except:
            return 0

class InteractiveMenu:
    def __init__(self):
        self.client = S2AGClient()
        self.current_release = None
        self.current_dataset = None
    
    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        if size_bytes == 0:
            return "Unknown size"
        
        size_names = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def clear_console(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_main_menu(self):
        """Display main menu and handle user input."""
        while True:
            self.clear_console()  # Clear console at the start of each iteration
            console.print(Panel.fit(
                "[bold cyan]Semantic Scholar Dataset Browser[/bold cyan]\n\n"
                f"Current Release: [green]{self.current_release or 'Not selected'}[/green]\n"
                f"Current Dataset: [green]{self.current_dataset or 'Not selected'}[/green]",
                title="Main Menu"
            ))
            
            options = [
                "Select Release",
                "Browse Datasets",
                "Dataset Details & Download Links",
                "Exit"
            ]
            
            table = Table(show_header=False, box=None)
            for i, option in enumerate(options, 1):
                table.add_row(f"[cyan]{i}.[/cyan]", option)
            
            console.print(table)
            
            choice = Prompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(options) + 1)])
            
            if choice == "1":
                self.select_release()
            elif choice == "2":
                if not self.current_release:
                    console.print("[yellow]Please select a release first[/yellow]")
                    console.input("\nPress Enter to continue...")
                    continue
                self.browse_datasets()
            elif choice == "3":
                if not self.current_dataset:
                    console.print("[yellow]Please select a dataset first[/yellow]")
                    console.input("\nPress Enter to continue...")
                    continue
                self.show_dataset_details()
            elif choice == "4":
                console.print("[cyan]Goodbye![/cyan]")
                sys.exit(0)

    def select_release(self):
        """Display release selection menu."""
        self.clear_console()  # Clear console before displaying releases
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Fetching releases...", total=None)
            releases = sorted(self.client.get_releases(), reverse=True)
        
        table = Table(title="Available Releases")
        table.add_column("Index", style="cyan")
        table.add_column("Release Date", style="green")
        table.add_column("Status", style="yellow")
        
        latest = max(releases)
        for i, release in enumerate(releases, 1):
            status = "Latest" if release == latest else ""
            table.add_row(str(i), release, status)
        
        console.print(table)
        
        choice = IntPrompt.ask(
            "Enter the index of the release",
            choices=[str(i) for i in range(1, len(releases) + 1)]
        )
        
        self.current_release = releases[choice - 1]

    def browse_datasets(self):
        """Display dataset selection menu."""
        self.clear_console()  # Clear console before displaying datasets
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Fetching datasets...", total=None)
            release_info = self.client.get_release_info(self.current_release)
        
        table = Table(title=f"Datasets in Release {self.current_release}")
        table.add_column("Index", style="cyan", width=5)
        table.add_column("Dataset Name", style="green", width=30)
        table.add_column("Description", width=80)  # Increased width for description
        
        for i, dataset in enumerate(release_info["datasets"], 1):
            description = dataset["description"]
            wrapped_description = Text(description, style="white", no_wrap=False)
            wrapped_description.truncate(300, overflow="ellipsis")  # Increased character limit
            table.add_row(str(i), dataset["name"], wrapped_description)
        
        console.print(table)
        
        choice = IntPrompt.ask(
            "Enter the index of the dataset",
            choices=[str(i) for i in range(1, len(release_info["datasets"]) + 1)]
        )
        
        self.current_dataset = release_info["datasets"][choice - 1]["name"]

    def show_dataset_details(self):
        """Display detailed dataset information and download links."""
        self.clear_console()  # Clear console before displaying dataset details
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Fetching dataset details...", total=None)
            dataset_info = self.client.get_dataset_info(self.current_release, self.current_dataset)
        
        # Calculate total size
        total_size = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Estimating dataset size...", total=len(dataset_info["files"]))
            for url in dataset_info["files"]:
                total_size += self.client.estimate_dataset_size(url)
                progress.advance(task)

        console.print(Panel(
            f"[cyan]Dataset:[/cyan] {dataset_info['name']}\n\n"
            f"[cyan]Description:[/cyan] {dataset_info['description']}\n\n"
            f"[cyan]Number of files:[/cyan] {len(dataset_info['files'])}\n\n"
            f"[cyan]Estimated total size:[/cyan] {self.format_size(total_size)}\n\n"
            f"[cyan]README excerpt:[/cyan]\n{dataset_info['README'][:500]}...",
            title="Dataset Details",
            expand=False
        ))
        
        console.print("\n[cyan]Download URLs:[/cyan]")
        for url in dataset_info["files"]:
            size = self.client.estimate_dataset_size(url)
            console.print(f"• {url} ({self.format_size(size)})")
        
        console.input("\nPress Enter to return to main menu...")

def main():
    try:
        menu = InteractiveMenu()
        menu.show_main_menu()
    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/cyan]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
