import logging
import urllib.parse
from pathlib import Path
from arxiv import get_arxiv_papers, extract_trending_topics_from_report, ArXivAPIError
from arxiv import store_pdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_url_encoding():
    """Test URL encoding in arXiv API calls"""
    query = "explainable OR classification"
    category = "cs.AI"
    
    # Build the search query string
    search_query = f"all:({query}) AND cat:{category}"
    
    # Encode parameters properly
    params = urllib.parse.urlencode({
        'search_query': search_query,
        'start': 0,
        'max_results': 2,  # Just get 2 results for testing
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    })
    
    # Print the properly encoded URL
    url = f"http://export.arxiv.org/api/query?{params}"
    print(f"Properly encoded URL: {url}")
    return url

def test_topic_extraction():
    """Test extracting topics from the trend report"""
    trend_reports = list(Path("../trend_cache").glob("trend_report_*.txt"))
    if trend_reports:
        latest_report = max(trend_reports, key=lambda p: p.stat().st_mtime)
        print(f"Using report: {latest_report}")
        
        # Test topic extraction
        topics = extract_trending_topics_from_report(latest_report)
        print(f"Extracted {len(topics)} topics")
        
        # Print first 2 topics for inspection
        for i, (topic_key, topic_info) in enumerate(list(topics.items())[:2]):
            print(f"\nTopic {i+1}: {topic_key}")
            print(f"  Description: {topic_info['description']}")
            print(f"  Query: {topic_info['query']}")
            print(f"  Category: {topic_info['category']}")
    else:
        print("No trend reports found")

def test_paper_fetch(max_results=5):
    """Test fetching a few papers from arXiv"""
    try:
        papers = get_arxiv_papers(
            query="explainable AI", 
            category="cs.AI",
            max_results=max_results,
            batch_size=max_results,
            pause_duration=5
        )
        
        print(f"\nSuccessfully fetched {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Published: {paper['published']}")
            print(f"   PDF URL: {paper['pdf']}")
            print()
            
    except ArXivAPIError as e:
        print(f"Error: {e}")

def test_store_pdf_function():
    """Test storing a paper PDF in the database"""
    print("\n4. Testing store_pdf function...")
    
    # Create a mock paper object that resembles what we'd get from ArXiv
    mock_paper = {
        "title": "Test Paper: Explainable AI Methods",
        "authors": ["Test Author"],
        "summary": "This is a test paper summary for testing the store_pdf function",
        "published": "2025-03-15",
        "updated": "2025-03-16",
        "arxiv_id": "test.12345",
        "pdf": "https://arxiv.org/pdf/test.12345",
        "pdf_path": "/tmp/test_paper.pdf",
        "categories": ["cs.AI"],
        "metadata": {
            "topic": "explainable_ai",
            "topic_description": "Explainable AI and Classification Techniques"
        }
    }
    
    # Create a test PDF file if needed
    from pathlib import Path
    test_pdf_path = Path("/tmp/test_paper.pdf")
    if not test_pdf_path.exists():
        with open(test_pdf_path, "wb") as f:
            f.write(b"%PDF-1.5\nTest PDF content")
        print(f"Created test PDF at {test_pdf_path}")
    
    # Update the mock paper with the actual path
    mock_paper["pdf_path"] = str(test_pdf_path)
    
    try:
        # First, inspect the function signature
        import inspect
        print(f"Store PDF function signature: {inspect.signature(store_pdf)}")
        
        # Try calling the function with positional arguments instead of keyword arguments
        # This is likely what the function expects
        result = store_pdf(mock_paper, "arxiv", mock_paper["title"])
        
        print(f"Store PDF result: {'Success' if result else 'Failed'}")
        print(f"Paper: {mock_paper['title']}")
        print(f"Source: arxiv")
        print(f"Topic: {mock_paper['metadata']['topic']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    #print("=== TESTING ARXIV SCRAPER ===\n")
    
    #print("1. Testing URL encoding...")
    #test_url_encoding()
    
    print("\n2. Testing topic extraction...")
    #test_topic_extraction()
    
    print("\n3. Testing paper fetch (limited to 5 papers)...")
    #test_paper_fetch(5)
    
    test_store_pdf_function()