from pydantic import BaseModel
from typing import List, Optional, Dict

class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: Optional[str] = ""
    authors: List[Dict] = []
    year: Optional[int] = None
    journal: Optional[str] = ""  # venue in S2
    url: Optional[str] = ""
    citation_count: Optional[int] = 0
    is_open_access: Optional[bool] = False
    fields_of_study: List[str] = []
    references: List[str] = []

    class Config:
        allow_population_by_dict = True

    @classmethod
    def from_s2_paper(cls, paper_data: dict):
        """Create a Paper instance from Semantic Scholar paper data."""
        return cls(
            corpus_id=paper_data.get('corpus_id'),
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract', ''),
            authors=paper_data.get('authors', []),
            year=paper_data.get('year'),
            journal=paper_data.get('venue', ''),
            url=paper_data.get('url', ''),
            citation_count=paper_data.get('citation_count', 0),
            fields_of_study=paper_data.get('fields_of_study', []),
            references=paper_data.get('references', []),
            text=paper_data.get('text', ''),
            content_source=paper_data.get('content_source', ''),
            pdf_hash=paper_data.get('pdf_hash', '')
        )

class PaperMetadata:
    def __init__(self, title: str, authors: list, year: int, journal: str, citation_count: int, influential_citation_count: int):
        self.title = title
        self.authors = authors
        self.year = year
        self.journal = journal
        self.citation_count = citation_count
        self.influential_citation_count = influential_citation_count
