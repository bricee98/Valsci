from pydantic import BaseModel
from typing import List, Optional, Dict

class Paper(BaseModel):
    id: str
    title: str
    authors: List[Dict[str, Optional[str]]]  # Changed from List[str] to List[Dict[str, Optional[str]]]
    year: Optional[int]
    journal: Optional[str]
    url: Optional[str]
    abstract: Optional[str]

class PaperMetadata:
    def __init__(self, title: str, authors: list, year: int, journal: str, citation_count: int, influential_citation_count: int):
        self.title = title
        self.authors = authors
        self.year = year
        self.journal = journal
        self.citation_count = citation_count
        self.influential_citation_count = influential_citation_count
