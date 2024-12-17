import logging
from typing import Dict, List
from app.models.paper import Paper
from app.services.openai_service import OpenAIService
from textwrap import dedent
from app.config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceScorer:
    def __init__(self):
        self.openai_service = OpenAIService()

    def calculate_paper_weight(self, paper: Paper) -> float:
        """Calculate the weight/reliability score for a paper."""
        try:
            # Get metrics
            max_h_index = self._get_max_author_h_index(paper.authors)
            citation_impact = self._calculate_citation_impact(paper)
            venue_impact = self._calculate_venue_impact(paper)
            
            # Normalize scores
            normalized_h_index = min(max_h_index / 100, 1.0)  # S2 h-indices can be higher than OpenAlex
            normalized_citation_impact = min(citation_impact / 1000, 1.0)
            normalized_venue_impact = min(venue_impact / 10, 1.0)
            
            # Calculate weighted average
            weights = {
                'author_impact': 0.4,
                'citation_impact': 0.4,
                'venue_impact': 0.2
            }
            
            score = (
                normalized_h_index * weights['author_impact'] +
                normalized_citation_impact * weights['citation_impact'] +
                normalized_venue_impact * weights['venue_impact']
            )
            
            logger.info(f"""
                Paper weight calculation for {paper.title}:
                - Max h-index: {max_h_index} (normalized: {normalized_h_index:.2f})
                - Citation impact: {citation_impact} (normalized: {normalized_citation_impact:.2f})
                - Venue impact: {venue_impact} (normalized: {normalized_venue_impact:.2f})
                - Final score: {score:.2f}
            """)
            
            return score

        except Exception as e:
            logger.error(f"Error calculating paper weight for {paper.title}: {str(e)}")
            return 0.0

    def _get_max_author_h_index(self, authors: List[Dict]) -> float:
        """Get the maximum h-index among the paper's authors."""
        try:
            h_indices = [
                author.get('hIndex', 0) 
                for author in authors 
                if isinstance(author.get('hIndex'), (int, float))
            ]
            return max(h_indices) if h_indices else 0
        except Exception as e:
            logger.error(f"Error getting max h-index: {str(e)}")
            return 0

    def _calculate_citation_impact(self, paper: Paper) -> float:
        """Calculate citation impact score."""
        try:
            # Get citation count
            citation_count = paper.citation_count or 0
            
            # Calculate years since publication
            current_year = 2024  # TODO: Get dynamically
            years_since_pub = max(1, current_year - (paper.year or current_year))
            
            # Calculate citations per year
            citations_per_year = citation_count / years_since_pub
            
            return citations_per_year
            
        except Exception as e:
            logger.error(f"Error calculating citation impact: {str(e)}")
            return 0

    def _calculate_venue_impact(self, paper: Paper) -> float:
        """Calculate venue impact using GPT to estimate journal/conference quality."""
        if not paper.journal:
            return 0.0

        try:
            system_prompt = """
            You are an expert in academic publishing and research venues. 
            Estimate the impact/prestige of an academic venue on a scale of 0-10.
            Consider factors like:
            - Venue reputation in the field
            - Publication standards and peer review
            - Typical citation rates
            - Publisher reputation
            
            Return only a number between 0 and 10.
            """

            prompt = f"""
            Rate the academic impact and prestige of this venue:
            Venue: {paper.journal}
            
            Return only a number between 0 and 10, where:
            0-2: Low impact or predatory venues
            3-5: Legitimate but lower impact venues
            6-8: Well-respected, mainstream venues
            9-10: Top venues in the field
            """

            score = self.openai_service.generate_number(prompt, system_prompt)
            return float(score)

        except Exception as e:
            logger.error(f"Error calculating venue impact for {paper.journal}: {str(e)}")
            return 0.0
