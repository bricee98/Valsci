import sqlite3
from pathlib import Path
import sys

def explore_index(release_id='latest'):
    # Find the latest release if not specified
    if release_id == 'latest':
        datasets_dir = Path('semantic_scholar/datasets')
        releases = [d.name for d in datasets_dir.iterdir() 
                   if d.is_dir() and d.name[0].isdigit()]
        if not releases:
            print("No releases found")
            return
        release_id = sorted(releases)[-1]
    
    db_path = Path(f'semantic_scholar/datasets/indices/{release_id}.db')
    if not db_path.exists():
        print(f"No index database found at {db_path}")
        return
        
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # List all tables
    print("\nTables in database:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        print(f"\nTable: {table[0]}")
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM paper_locations")
    total = cursor.fetchone()[0]
    print(f"\nTotal index entries: {total:,}")
    
    # Get counts by dataset
    print("\nEntries by dataset:")
    cursor.execute("""
        SELECT dataset, COUNT(*) as count 
        FROM paper_locations 
        GROUP BY dataset
    """)
    for dataset, count in cursor.fetchall():
        print(f"{dataset}: {count:,}")
    
    # Sample some entries
    print("\nSample entries:")
    cursor.execute("""
        SELECT id, id_type, dataset, file_path, line_offset 
        FROM paper_locations 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"ID: {row[0]}")
        print(f"Type: {row[1]}")
        print(f"Dataset: {row[2]}")
        print(f"File: {Path(row[3]).name}")
        print(f"Offset: {row[4]}")
        print()
    
    # Interactive query mode
    while True:
        print("\nEnter a paper ID to look up (or 'q' to quit):")
        paper_id = input().strip()
        if paper_id.lower() == 'q':
            break
            
        cursor.execute("""
            SELECT dataset, file_path, line_offset 
            FROM paper_locations 
            WHERE id = ?
        """, (paper_id,))
        results = cursor.fetchall()
        
        if not results:
            print("Paper ID not found in index")
        else:
            for dataset, file_path, offset in results:
                print(f"\nFound in {dataset}")
                print(f"File: {Path(file_path).name}")
                print(f"Offset: {offset}")

if __name__ == '__main__':
    release_id = sys.argv[1] if len(sys.argv) > 1 else 'latest'
    explore_index(release_id) 