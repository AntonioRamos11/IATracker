import feedparser
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict
import logging
from datetime import datetime

from storedata import store_pdf
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArXivAPIError(Exception):
    """Custom exception for arXiv API errors"""
    pass

def create_session() -> requests.Session:
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'AITracker/1.0 (contact@example.com)',
        'Accept': 'application/xml'
    })
    return session

def validate_arxiv_response(feed: feedparser.FeedParserDict) -> None:
    """Validate the arXiv API response structure"""
    if feed.get('bozo', 0) != 0:
        raise ArXivAPIError(f"Feed parsing error: {feed.bozo_exception}")
    
    if not hasattr(feed, 'entries'):
        raise ArXivAPIError("Invalid response format - missing entries")

def construct_pdf_url(arxiv_id: str) -> str:
    """Safely construct PDF URL from arXiv ID"""
    base_url = "https://arxiv.org/pdf/"
    return f"{base_url}{arxiv_id}.pdf"

def parse_arxiv_entry(entry) -> Dict:
    """Parse individual arXiv entry with error handling"""
    try:
        arxiv_id = entry.id.split('/abs/')[-1]
        return {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary,
            "published": datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ'),
            "link": entry.link,
            "pdf": construct_pdf_url(arxiv_id),
            "arxiv_id": arxiv_id,
            "doi": entry.get('arxiv_doi', '')
        }
    except (AttributeError, KeyError, ValueError) as e:
        logger.error(f"Failed to parse entry: {e}")
        return None


def get_arxiv_papers(
    query: str = "artificial intelligence",
    max_results: int = 100,
    category: str = "cs.AI"
) -> List[Dict]:
    """
    Fetch papers from arXiv API with enhanced error handling and retries
    
    Args:
        query: Search query string
        max_results: Number of results to return (1-1000)
        category: arXiv category filter (default: cs.AI)
    
    Returns:
        List of paper dictionaries
    """
    try:
        # Validate input parameters
        max_results = max(1, min(max_results, 1000))
        
        # Build API URL with parameters
        params = {
            'search_query': f'all:{query} cat:{category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        # Make API request 
        session = create_session()
        response = session.get(
            "http://export.arxiv.org/api/query",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        # Parse and validate feed
        feed = feedparser.parse(response.content)
        validate_arxiv_response(feed)
        
        # Process entries
        papers = []
        for entry in feed.entries:
            parsed = parse_arxiv_entry(entry)
            if parsed:
                pdf_path = store_pdf(parsed['pdf'], 'arxiv', parsed['title'])
                parsed['pdf_path'] = pdf_path
                papers.append(parsed)
        
        logger.info(f"Successfully fetched {len(papers)} papers")
        return papers
    
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error occurred: {e}")
        raise ArXivAPIError(f"API request failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ArXivAPIError(f"Failed to fetch papers: {e}") from e
# Example usage
if __name__ == "__main__":
    try:
        max_results = 100
        papers = get_arxiv_papers(max_results)   
        for idx, paper in enumerate(papers[:max_results], 1):
            print(f"{idx}. {paper['title']}")
            print(f"   PDF: {paper['pdf']}\n")
    except ArXivAPIError as e:
        print(f"Error fetching papers: {e}")