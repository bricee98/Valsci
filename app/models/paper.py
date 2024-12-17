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

    class Config:
        allow_population_by_dict = True

    @classmethod
    def from_s2_paper(cls, s2_paper: Dict) -> 'Paper':
        """Create a Paper instance from Semantic Scholar paper data."""
        return cls(
            paper_id=s2_paper['paperId'],
            title=s2_paper.get('title', ''),
            abstract=s2_paper.get('abstract', ''),
            authors=s2_paper.get('authors', []),
            year=s2_paper.get('year'),
            journal=s2_paper.get('venue', ''),
            url=s2_paper.get('url', ''),
            citation_count=s2_paper.get('citation_count', 0),
            is_open_access=s2_paper.get('isOpenAccess', False),
            fields_of_study=s2_paper.get('fieldsOfStudy', [])
        )

class PaperMetadata:
    def __init__(self, title: str, authors: list, year: int, journal: str, citation_count: int, influential_citation_count: int):
        self.title = title
        self.authors = authors
        self.year = year
        self.journal = journal
        self.citation_count = citation_count
        self.influential_citation_count = influential_citation_count
