import feedparser
import time
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict
import logging
from datetime import datetime
import socks
from stem.control import Controller
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
6
class ArXivAPIError(Exception):
    """Custom exception for arXiv API errors"""
    pass



def create_session() -> requests.Session:
    """Create a requests session with realistic headers to avoid blocks."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429],  # 429 = Too Many Requests
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/pdf',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://arxiv.org/',
        'Connection': 'keep-alive',
    })

    # Add cookies to the session
    cookies = [
        {
            "name": "browser",
            "value": "187.188.34.209.1732737085204374",
            "domain": ".arxiv.org",
            "path": "/",
            "secure": False
        },
        {
            "name": "captchaAuth",
            "value": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyN2Q3Zjg2ZC1lM2IxLTQ5NGQtOTk4Mi02OGI2N2JlZmIzN2UiLCJleHAiOjE3NDA1NTIyNzQsImlwIjoxODcuMTg4LjM0LjIwOSwiaWF0IjoxNzQwNTUwNDc1LCJpc3MiOiJGYXN0bHkifQ.MHhhNDJkZThiMzljYTc1ODI0N2I0NWU5OWFhNzYwMmU5NTQ2YmZjZmUyODEyYTg0ZWE3MWU2ZDEzYThhMzExZGRh",
            "domain": "arxiv.org",
            "path": "/",
            "secure": False
        }
    ]

    for cookie in cookies:
        session.cookies.set(**cookie)

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
    url = f"{base_url}{arxiv_id}.pdf"
    
    print(f"Generated PDF URL: {url}")  # Debugging
    return url

import random
def smart_delay():
    delay = random.uniform(10, 20)  # Random delay between 10-20 seconds
    print(f"Sleeping for {delay:.2f} seconds to avoid detection...")
    time.sleep(delay)

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
    max_results: int = 1000,
    category: str = "cs.AI",
    batch_size: int = 10,
    pause_duration: int = 30
) -> List[Dict]:
    """
    Fetch papers from arXiv API with enhanced error handling and retries
    
    Args:
        query: Search query string
        max_results: Number of results to return (1-1000)
        category: arXiv category filter (default: cs.AI)
        batch_size: Number of results to fetch per batch
        pause_duration: Duration to pause between batches (in seconds)
    
    Returns:
        List of paper dictionaries
    """
    try:
        # Validate input parameters
        max_results = max(1, min(max_results, 1000))
        batch_size = max(1, min(batch_size, max_results))
        
        papers = []
        start = 0
        
        while start < max_results:
            # Build API URL with parameters
            params = {
                'search_query': f'all:{query} cat:{category}',
                'start': start,
                'max_results': batch_size,
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
            for entry in feed.entries:
                parsed = parse_arxiv_entry(entry)
                if parsed:
                    papers.append(parsed)
            
            logger.info(f"Successfully fetched {len(feed.entries)} papers in batch starting at {start}")
            
            # Update start for next batch
            start += batch_size
            
            # Pause between batches
            #wait from a enter from a input
            input("Press Enter to continue...")
                        
        
        logger.info(f"Successfully fetched a total of {len(papers)} papers")
        return papers
    
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error occurred: {e}")
        raise ArXivAPIError(f"API request failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ArXivAPIError(f"Failed to fetch papers: {e}") from e

def save_pdf_links(papers: List[Dict], filename: str) -> None:
    """Save PDF links to a file for manual download"""
    with open(filename, 'w') as f:
        for paper in papers:
            f.write(f"{paper['pdf']}\n")
    logger.info(f"Saved PDF links to {filename}")

# Example usage
if __name__ == "__main__":
    try:
        max_results = 1000
        papers = get_arxiv_papers(max_results=max_results, batch_size=5, pause_duration=30)
        save_pdf_links(papers, "pdf_links.txt")
        for idx, paper in enumerate(papers[:max_results], 1):
            print(f"{idx}. {paper['title']}")
            print(f"   PDF: {paper['pdf']}\n")
    except ArXivAPIError as e:
        print(f"Error fetching papers: {e}")