import logging
from typing import Dict, List
from app.models.paper import Paper
from textwrap import dedent
from app.config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceScorer:

    async def calculate_paper_weight(self, processed_paper, ai_service, bibliometric_config=None) -> float:
        """Calculate the weight/reliability score for a paper.
        
        Args:
            processed_paper: The paper to score
            ai_service: Service for AI operations
            bibliometric_config: Optional configuration for bibliometric scoring
                {
                    'use_bibliometrics': bool,
                    'author_impact_weight': float,
                    'citation_impact_weight': float,
                    'venue_impact_weight': float
                }
        """
        try:
            # Check if bibliometrics should be used
            if bibliometric_config is not None and not bibliometric_config.get('use_bibliometrics', True):
                # Return neutral score if bibliometrics are disabled
                return 0.5, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}
            
            # Add validation for processed_paper
            if processed_paper is None:
                logger.error("processed_paper is None")
                return 0.0, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}
            
            # Add validation for paper key
            if 'paper' not in processed_paper:
                logger.error(f"No 'paper' key in processed_paper: {processed_paper}")
                return 0.0, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}

            # Get the nested paper data
            paper = processed_paper['paper']
            
            # Add validation for paper object
            if paper is None:
                logger.error("paper object is None")
                return 0.0, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}
            
            # Log the paper structure for debugging
            logger.debug(f"Processing paper structure: {paper}")
            
            # Get metrics
            avg_h_index = self._get_author_h_index(paper.get('authors', []))
            citation_impact = self._calculate_citation_impact(paper)
            venue_impact, usage = await self._calculate_venue_impact(paper.get('venue'), ai_service)
            
            # Normalize scores
            normalized_h_index = min(avg_h_index / 50, 1.0)  # Adjusted threshold since we're using average
            normalized_citation_impact = min(citation_impact / 1000, 1.0)
            normalized_venue_impact = min(venue_impact / 10, 1.0)
            
            # Calculate weighted average with default or provided weights
            weights = {
                'author_impact': 0.4,
                'citation_impact': 0.4,
                'venue_impact': 0.2
            }
            
            # Override with custom weights if provided
            if bibliometric_config:
                if 'author_impact_weight' in bibliometric_config:
                    weights['author_impact'] = bibliometric_config['author_impact_weight']
                if 'citation_impact_weight' in bibliometric_config:
                    weights['citation_impact'] = bibliometric_config['citation_impact_weight']
                if 'venue_impact_weight' in bibliometric_config:
                    weights['venue_impact'] = bibliometric_config['venue_impact_weight']
            
            # Normalize weights to ensure they sum to 1
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                weights = {k: v/weight_sum for k, v in weights.items()}
            
            score = (
                normalized_h_index * weights['author_impact'] +
                normalized_citation_impact * weights['citation_impact'] +
                normalized_venue_impact * weights['venue_impact']
            )
            
            logger.info(f"""
                Paper weight calculation for {paper.get('title', 'Unknown Title')}:
                - Average h-index: {avg_h_index} (normalized: {normalized_h_index:.2f})
                - Citation impact: {citation_impact} (normalized: {normalized_citation_impact:.2f})
                - Venue impact: {venue_impact} (normalized: {normalized_venue_impact:.2f})
                - Final score: {score:.2f}
                - Weights used: {weights}
            """)
            
            return score, usage

        except Exception as e:
            logger.error(f"Error calculating paper weight: {str(e)}")
            return 0.0, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}

    def _get_author_h_index(self, authors: List[Dict]) -> float:
        """Get the average h-index among the paper's authors (first and last only)."""
        try:
            h_indices = [
                author.get('hIndex', 0) 
                for author in authors 
                if isinstance(author.get('hIndex'), (int, float))
            ]
            return sum(h_indices) / len(h_indices) if h_indices else 0
        except Exception as e:
            logger.error(f"Error getting average h-index: {str(e)}")
            return 0

    def _calculate_citation_impact(self, paper: Dict) -> float:
        """Calculate citation impact score."""
        try:
            # Get citation count
            citation_count = paper.get('citationCount', 0)
            
            # Calculate years since publication
            current_year = 2024  # TODO: Get dynamically
            years_since_pub = max(1, current_year - (paper.get('year', current_year) or current_year))
            
            # Calculate citations per year
            citations_per_year = citation_count / years_since_pub
            
            return citations_per_year
            
        except Exception as e:
            logger.error(f"Error calculating citation impact: {str(e)}")
            return 0

    async def _calculate_venue_impact(self, paper_journal, ai_service) -> float:
        """Calculate venue impact using GPT to estimate journal/conference quality."""
        if not paper_journal:
            # Return tuple of (score, usage) instead of just score
            return 0.0, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}

        try:
            system_prompt = """
            You are an expert in academic publishing and research venues. 
            Estimate the impact/prestige of an academic venue on a scale of 0-10.
            Consider factors like:
            - Venue reputation in the field
            - Publication standards and peer review
            - Typical citation rates
            - Publisher reputation
            
            Return only json object with a single key "score" and a number between 0 and 10.
            """

            prompt = f"""
            Rate the academic impact and prestige of this venue:
            Venue: {paper_journal}
            
            Return only json object with a single key "score" and a number between 0 and 10, where:
            0-2: Low impact or predatory venues
            3-5: Legitimate but lower impact venues
            6-8: Well-respected, mainstream venues
            9-10: Top venues in the field
            """

            result = await ai_service.generate_json_async(prompt, system_prompt)
            response = result['content']
            usage = result['usage']
            score = response.get('score', 0)

            return float(score), usage

        except Exception as e:
            logger.error(f"Error calculating venue impact for {paper_journal}: {str(e)}")
            # Return tuple of (score, usage) instead of just score
            return 0.0, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0}
