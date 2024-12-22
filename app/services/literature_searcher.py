from typing import List, Union, Tuple
import logging
from app.models.claim import Claim
from app.models.paper import Paper
from app.config.settings import Config
from semantic_scholar.utils.searcher import S2Searcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiteratureSearcher:
    def __init__(self):
        self.email = Config.USER_EMAIL
        self.s2_searcher = S2Searcher()
        self.saved_search_queries = []

    def search_papers(self, claim: Claim) -> List[Paper]:
        """Search for papers relevant to the claim."""
        try:
            # Get search configuration from claim
            num_queries = claim.search_config.get('numQueries', 5)
            results_per_query = claim.search_config.get('resultsPerQuery', 5)
            
            # Search using S2 searcher
            raw_papers = self.s2_searcher.search_papers_for_claim(
                claim.text, 
                num_queries=num_queries,
                results_per_query=results_per_query
            )
            
            # Save search queries for reporting
            self.saved_search_queries = self.s2_searcher.saved_search_queries
            
            # Convert to Paper objects with error handling
            papers = []
            for raw_paper in raw_papers:
                print("raw_paper", raw_paper)
                try:
                    # Ensure fields_of_study is a list
                    if raw_paper.get('fields_of_study') is None:
                        raw_paper['fields_of_study'] = []
                    papers.append(Paper.from_s2_paper(raw_paper))
                except Exception as e:
                    logger.error(f"Error converting paper {raw_paper.get('paper_id')}: {str(e)}")
                    continue
            
            # Sort by citation count (most cited first)
            papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return []

    def fetch_paper_content(self, paper: Paper, claim: Claim) -> Tuple[str, str]:
        """Fetch the full text content of a paper."""
        try:
            if not paper.corpus_id:
                logger.warning(f"No corpus ID available for paper {paper.paper_id}")
                return paper.abstract or "", "abstract_only"

            logger.info(f"Fetching content for corpus ID: {paper.corpus_id}")
            content = self.s2_searcher.get_paper_content(str(paper.corpus_id))
            
            if content:
                logger.info(f"Successfully retrieved {content['source']} content for corpus ID: {paper.corpus_id}")
                
                # If we got just an abstract but the paper has a longer one, use the longer version
                if (content['source'] == 'abstract' and 
                    paper.abstract and 
                    len(paper.abstract) > len(content['text'])):
                    return paper.abstract, "abstract_only"
                    
                return content['text'], content['source']
            
            # Fallback to paper's abstract if no content found
            logger.warning(f"No content found for corpus ID: {paper.corpus_id}, falling back to abstract")
            return paper.abstract or "", "abstract_only"
                
        except Exception as e:
            logger.error(f"Error fetching paper content for {paper.corpus_id}: {str(e)}")
            return paper.abstract or "", "abstract_only"

