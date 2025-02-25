from scholarly import scholarly
from typing import List, Dict, Optional
import logging
import random
import time
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleScholarError(Exception):
    """Custom exception for Google Scholar errors"""
    pass

class GoogleScholarConfig:
    """Configuration constants for scraping safety"""
    MAX_RESULTS = 50
    MIN_DELAY = 2
    MAX_RETRIES = 3
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    ]

def get_random_delay():
    """Generate random delay between requests"""
    return random.uniform(GoogleScholarConfig.MIN_DELAY, GoogleScholarConfig.MAX_DELAY)

def handle_ratelimit(response: requests.Response):
    """Handle potential rate limiting"""
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        logger.warning(f"Rate limited. Sleeping for {retry_after} seconds")
        time.sleep(retry_after)
        return True
    return False

def parse_scholar_entry(entry: dict) -> Optional[Dict]:
    """Parse and validate a Google Scholar entry"""
    try:
        bib = entry.get('bib', {})
        return {
            "title": bib.get('title', 'Untitled'),
            "url": entry.get('pub_url', ''),
            "abstract": bib.get('abstract', ''),
            "year": bib.get('pub_year', None),
            "authors": bib.get('author', []),
            "citations": entry.get('num_citations', 0),
            "publication_date": parse_publication_date(bib),
            "scholar_id": entry.get('author_id', [])
        }
    except Exception as e:
        logger.error(f"Error parsing entry: {e}")
        return None

def parse_publication_date(bib: dict) -> Optional[datetime]:
    """Attempt to parse publication date from bib data"""
    try:
        year = int(bib.get('pub_year', datetime.now().year))
        return datetime(year=year, month=1, day=1)
    except (ValueError, TypeError):
        return None

def get_google_scholar_papers(
    query: str = "artificial intelligence",
    num_papers: int = 5,
    timeout: int = 10
) -> List[Dict]:
    """
    Robust Google Scholar paper fetcher with safety measures
    
    Args:
        query: Search query string
        num_papers: Number of results (1-50)
        timeout: Request timeout in seconds
    
    Returns:
        List of parsed paper dictionaries
    """
    num_papers = max(1, min(num_papers, GoogleScholarConfig.MAX_RESULTS))
    papers = []
    retries = 0
    
    try:
        scholarly.use_proxy = True
        scholarly.headers = {
            'User-Agent': random.choice(GoogleScholarConfig.USER_AGENTS)
        }
        
        search_query = scholarly.search_pubs(query)
        
        while len(papers) < num_papers and retries < GoogleScholarConfig.MAX_RETRIES:
            try:
                time.sleep(get_random_delay())
                paper = next(search_query)
                parsed = parse_scholar_entry(paper)
                
                if parsed:
                    papers.append(parsed)
                    logger.debug(f"Collected paper: {parsed['title']}")
            except StopIteration:
                logger.info("Reached end of search results")
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error: {e}")
                retries += 1
                if handle_ratelimit(e.response):
                    continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                retries += 1
                time.sleep(get_random_delay() * 2)
        
        if retries >= GoogleScholarConfig.MAX_RETRIES:
            logger.error("Max retries reached. Returning partial results")
            
        return papers[:num_papers]
    
    except Exception as e:
        logger.error(f"Critical error in Google Scholar fetch: {e}")
        raise GoogleScholarError(f"Failed to fetch papers: {e}") from e

# Example usage
if __name__ == "__main__":
    try:
        papers = get_google_scholar_papers(
            query="machine learning",
            num_papers=3
        )
        for idx, paper in enumerate(papers, 1):
            print(f"{idx}. {paper['title']}")
            print(f"   Year: {paper.get('year', 'Unknown')}")
            print(f"   URL: {paper['url']}\n")
    except GoogleScholarError as e:
        print(f"Error fetching papers: {e}")