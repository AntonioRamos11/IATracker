import feedparser,time
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
import hashlib
import re
import pathlib
from ArXivDownloader import ArXivDownloader

from storedata import store_pdf
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArXivAPIError(Exception):
    """Custom exception for arXiv API errors"""
    pass

"""def connect_tor():
    Switch to a new Tor identity to change IP.
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password="your-tor-password")
        controller.signal(2)  # New identity

socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 9050)
socket.socket = socks.socksocket"""

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
    """
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/pdf',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://arxiv.org/',
        'Connection': 'keep-alive',
    })"""
    session.headers.update(get_random_headers())

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
    delay = random.uniform(10, 20)  # Random delay between 3-10 seconds
    print(f"Sleeping for {delay:.2f} seconds to avoid detection...")
    time.sleep(delay)


from fake_useragent import UserAgent
import requests

def get_random_headers():
    ua = UserAgent()
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://arxiv.org/',
        'DNT': '1'  # Do Not Track
    }

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

import undetected_chromedriver as uc

def solve_captcha_manually(pdf_url):
    """Opens the URL in a real browser to solve the CAPTCHA manually."""
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)

    driver.get(pdf_url)
    print("Solve the CAPTCHA manually, then press Enter in the terminal...")
    input()  # Wait for user confirmation

    final_url = driver.current_url
    driver.quit()

    logger.info(f"Final URL after solving CAPTCHA: {final_url}")
    
    return final_url if final_url.endswith(".pdf") else None
def get_arxiv_papers(
    query: str = "robotics with AI",
    max_results: int = 300,
    category: str = "cs.AI",
    batch_size: int = 10,
    pause_duration: int = 30,
    max_retries: int = 3
) -> List[Dict]:
    """
    Fetch papers from arXiv API with enhanced error handling and retries
    
    Args:
        query: Search query string
        max_results: Number of results to return (1-1000)
        category: arXiv category filter (default: cs.AI)
        batch_size: Number of results to fetch per batch
        pause_duration: Duration to pause between batches (in seconds)
        max_retries: Maximum number of retries for fetching a PDF
    
    Returns:
        List of paper dictionaries
    """
    try:
        # Validate input parameters
        max_results = max(1, min(max_results, 1000))
        batch_size = max(1, min(batch_size, max_results))
        
        papers = []
        start = 0
        downloader = ArXivDownloader()
        
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
                    retries = 0
                    while retries < max_retries:
                        pdf_content = downloader.download_pdf(parsed['arxiv_id'])
                        if pdf_content:
                            # Save the PDF content to a file
                            file_hash = hashlib.sha256(pdf_content).hexdigest()[:16]
                            clean_title = re.sub(r'[^\w\s-]', '', parsed['title']).strip().replace(' ', '_').replace('\n', '')
                            save_path = pathlib.Path(f"papers/arxiv/{clean_title}_{file_hash}.pdf")
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(save_path, "wb") as f:
                                f.write(pdf_content)
                            parsed['pdf_path'] = str(save_path)
                            papers.append(parsed)
                            break
                        else:
                            retries += 1
                            logger.warning(f"Retrying {parsed['title']} ({retries}/{max_retries})")
                            time.sleep(5)  # Wait before retrying
                    if retries == max_retries:
                        logger.error(f"Failed to fetch {parsed['title']} after {max_retries} retries")
            
            logger.info(f"Successfully fetched {len(feed.entries)} papers in batch starting at {start}")
            
            # Update start for next batch
            start += batch_size
            
            # Pause between batches
            if start < max_results:
                logger.info(f"Pausing for {pause_duration} seconds to avoid getting banned")
                smart_delay()
        
        logger.info(f"Successfully fetched a total of {len(papers)} papers")
        return papers
    
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error occurred: {e}")
        raise ArXivAPIError(f"API request failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ArXivAPIError(f"Failed to fetch papers: {e}") from e
    
from Topic_Refiner import TopicRefiner
    
def extract_trending_topics_from_report(report_path):
    """
    Extract trending topics from the topic analysis report with improved names.
    
    Args:
        report_path: Path to the trend report file
    
    Returns:
        Dictionary of trending topics suitable for arXiv queries
    """
    # Initialize trending topics dictionary
    trending_topics = {}
    
    try:
        # Use your existing TopicRefiner to get better topic names
        refiner = TopicRefiner()
        refined_topics = refiner.refine_topics_from_report(report_path)
        
        # Convert refined topics to the format needed for ArXiv searches
        for topic in refined_topics:
            topic_id = topic['original_id']
            
            # Create a clean key from the refined title
            clean_title = topic['refined_title'].lower().replace(' ', '_').replace('&', 'and')
            topic_key = f"topic_{topic_id}_{clean_title}"
            
            # Get top terms for the query
            top_terms = topic['top_terms'][:4]  # Use top 4 terms
            
            # Determine appropriate category based on refined title
            category = "cs.AI"  # Default
            if any(term in topic['refined_title'].lower() for term in ["language", "attention", "transformer"]):
                category = "cs.CL"
            elif any(term in topic['refined_title'].lower() for term in ["graph", "network"]):
                category = "cs.LG"
            elif any(term in topic['refined_title'].lower() for term in ["quantum"]):
                category = "quant-ph"
            elif any(term in topic['refined_title'].lower() for term in ["retrieval", "dense"]):
                category = "cs.IR"
            elif any(term in topic['refined_title'].lower() for term in ["time", "series", "forecast"]):
                category = "cs.LG"
            
            # Create topic entry with improved representation
            trending_topics[topic_key] = {
                "query": " OR ".join(top_terms),  # Using OR for broader results
                "category": category,
                "description": topic['refined_title'],
                "original_terms": top_terms  # Keep original terms for reference
            }
            
        print(f"Extracted {len(trending_topics)} trending topics with refined names")
        return trending_topics
        
    except Exception as e:
        print(f"Error extracting topics from report: {e}")
        # Return default topics as fallback
        return {
            "explainable_ai": {
                "query": "explainable OR classification OR deep",
                "category": "cs.AI",
                "description": "Explainable AI and Classification Techniques"
            },
            # Add other default topics...
        }

import re
import json
from pathlib import Path
def scrape_trending_topics(max_results_per_topic=100, batch_size=10):
    """
    Scrape papers from trending topics identified in the topic analysis.
    """
    # Try to load topics from the latest trend report
    trend_reports = list(Path("./trend_cache").glob("trend_report_*.txt"))
    if trend_reports:
        # Get the most recent report
        latest_report = max(trend_reports, key=lambda p: p.stat().st_mtime)
        print(f"Using report: {latest_report.name}")
        
        # Extract topics with our improved TopicRefiner approach
        trending_topics = extract_trending_topics_from_report(latest_report)
    else:
        # Use default topics if no report is available
        print("No trend report found, using default topics")
        trending_topics = {
            # Default topics...
        }
    
    all_papers = {}
    
    for topic_name, topic_info in trending_topics.items():
        print(f"\nScraping papers for trend: {topic_info['description']}...")
        print(f"Query: {topic_info['query']}, Category: {topic_info['category']}")
        
        try:
            # Use the existing get_arxiv_papers function with separate query and category
            papers = get_arxiv_papers(
                query=topic_info['query'],
                category=topic_info['category'],
                max_results=max_results_per_topic,
                batch_size=batch_size,
                pause_duration=30
            )
            
            # Add topic metadata to each paper
            for paper in papers:
                if 'metadata' not in paper:
                    paper['metadata'] = {}
                paper['metadata']['topic'] = topic_name
                paper['metadata']['topic_description'] = topic_info['description']
            
            print(f"Successfully fetched {len(papers)} papers for {topic_info['description']}")
            all_papers[topic_name] = papers
            
        except Exception as e:
            logger.error(f"Failed to fetch papers for {topic_name}: {e}")
    
    return all_papers

def analyze_topic_coverage(all_papers):
    """
    Analyze the coverage of papers across trending topics.
    
    Args:
        all_papers: Dictionary of papers by topic
        
    Returns:
        Summary of paper coverage by topic
    """
    coverage = {}
    
    for topic, papers in all_papers.items():
        # Count papers by year
        years = {}
        for paper in papers:
            year = paper['published'].year
            years[year] = years.get(year, 0) + 1
        
        coverage[topic] = {
            "total": len(papers),
            "by_year": years,
            "sample_titles": [p["title"] for p in papers[:3]]
        }
    
    return coverage

if __name__ == "__main__":
    try:
        # Choose between general search or trend-focused search
        search_mode = "trends"  # or "general"
        
        if search_mode == "general":
            max_results = 100
            papers = get_arxiv_papers(
                query="artificial intelligence", 
                max_results=max_results, 
                batch_size=5, 
                pause_duration=30
            )
            
            for idx, paper in enumerate(papers[:max_results], 1):
                print(f"{idx}. {paper['title']}")
                print(f"   PDF: {paper['pdf']}\n")
                
        elif search_mode == "trends":
            # Use the trend-based search
            trend_papers = scrape_trending_topics(
                max_results_per_topic=50,  # 50 papers per topic
                batch_size=5              # Fetch in batches of 5
            )
            
            # Analyze the coverage
            coverage = analyze_topic_coverage(trend_papers)
            
            # Print a summary
            print("\n===== TREND COVERAGE SUMMARY =====")
            for topic, stats in coverage.items():
                print(f"\n{topic.upper()} - Total: {stats['total']} papers")
                print(f"  Topic: {stats.get('description', 'Unknown')}")
                if stats['by_year']:
                    print(f"  Years: {', '.join([f'{y}: {c}' for y, c in stats['by_year'].items()])}")
                print("  Sample titles:")
                for title in stats['sample_titles']:
                    print(f"    - {title}")
            
            # Save all papers to database
            saved_count = 0
            for topic, papers in trend_papers.items():
                for paper in papers:
                    if "pdf_path" in paper:  # If paper was successfully downloaded
                        # Add topic tag to metadata
                        paper["metadata"] = paper.get("metadata", {})
                        paper["metadata"]["topic"] = topic
                        
                        # Fix: Add the missing parameters to store_pdf
                        store_pdf(
                            paper=paper,
                            source="arxiv",
                            title=paper.get("title", "Untitled Paper")
                        )
                        saved_count += 1
            
            print(f"\nSaved {saved_count} papers to database with topic tags")
            
    except ArXivAPIError as e:
        print(f"Error fetching papers: {e}")
