# Add these imports at the top of the file
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict
import logging
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure TopicProcessor is available for external imports
try:
    from Scrapers.topic_processor import TopicProcessor
except ImportError:
    # When this module is imported from the same directory
    try:
        from topic_processor import TopicProcessor
    except ImportError:
        logger.error("Could not import TopicProcessor class")

# Download required NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
# Configure logger

class TrendAnalyzer:
    """Advanced trend analysis system for arXiv papers"""
    """Advanced trend analysis system for arXiv papers"""
    
    def __init__(self):
        """Initialize the trend analyzer"""
        self.stop_words = set(stopwords.words('english'))
        self.custom_stops = {
            'using', 'based', 'paper', 'method', 'show', 'approach', 'result', 'also',
            'propose', 'use', 'new', 'via', 'model', 'two', 'one', 'first', 'novel',
            'can', 'multiple', 'data', 'study', 'research', 'problem', 'different'
        }
        self.stop_words.update(self.custom_stops)
        
    def clean_text(self, text):
        """Clean text by removing special characters and stopwords"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in text.split() if word not in self.stop_words and len(word) > 2]
        return tokens
    
    def extract_ngrams(self, text_list, min_n=2, max_n=3):
        """Extract n-grams from a list of text strings"""
        all_ngrams = []
        
        for text in text_list:
            tokens = self.clean_text(text)
            # Skip very short texts
            if len(tokens) < min_n:
                continue
                
            # Extract different n-gram sizes
            for n in range(min_n, min(max_n + 1, len(tokens) + 1)):
                text_ngrams = list(ngrams(tokens, n))
                all_ngrams.extend([" ".join(ng) for ng in text_ngrams])
                
        return Counter(all_ngrams)
    
    def analyze_papers_by_time(self, papers, months_back=3):
        """Analyze papers by time period to identify emerging trends"""
        now = datetime.now()
        cutoff_date = now - timedelta(days=30 * months_back)
        
        # Split papers into recent and older
        recent_papers = []
        older_papers = []
        
        for paper in papers:
            pub_date = paper.get('published', None)
            if not pub_date:
                continue
                
            if isinstance(pub_date, datetime):
                if pub_date > cutoff_date:
                    recent_papers.append(paper)
                else:
                    older_papers.append(paper)
                    
        # Extract text for analysis
        recent_texts = [f"{p.get('title', '')} {p.get('summary', '')}" for p in recent_papers]
        older_texts = [f"{p.get('title', '')} {p.get('summary', '')}" for p in older_papers]
        
        # Get n-gram frequencies
        recent_ngrams = self.extract_ngrams(recent_texts)
        older_ngrams = self.extract_ngrams(older_texts)
        
        # Calculate growth ratio for each n-gram
        trending_topics = []
        for ngram, recent_count in recent_ngrams.most_common(100):
            # Normalize by number of papers
            recent_ratio = recent_count / max(len(recent_papers), 1)
            older_count = older_ngrams.get(ngram, 0)
            older_ratio = older_count / max(len(older_papers), 1)
            
            # Calculate growth (avoid division by zero)
            if older_ratio > 0:
                growth = recent_ratio / older_ratio
            else:
                growth = recent_ratio * 2  # New topics get a boost
                
            # Only include topics with meaningful presence
            if recent_count >= 3:
                trending_topics.append({
                    "term": ngram,
                    "recent_count": recent_count,
                    "older_count": older_count,
                    "growth": growth,
                    "score": growth * recent_count  # Combined score
                })
                
        # Sort by combined score
        trending_topics.sort(key=lambda x: x["score"], reverse=True)
        return trending_topics
        
    def generate_search_queries(self, trending_topics, top_n=5):
        """Generate optimized search queries from trending topics"""
        queries = []
        for topic in trending_topics[:top_n]:
            # Build a query with the topic term as the main focus
            term = topic['term']
            query = f'(ti:"{term}" OR abs:"{term}") AND submittedDate:[2024 TO 2025]'
            queries.append({
                "term": term,
                "query": query,
                "score": topic['score'],
                "growth": topic['growth']
            })
        return queries

def identify_emerging_trends(
    categories=None, 
    initial_sample_size=1000,
    months_to_analyze=3,
    include_citations=False
) -> Dict:
    """
    Identify emerging research trends in arXiv by analyzing recent papers.
    
    Args:
        categories: List of arXiv categories to analyze (default: major CS and AI categories)
        initial_sample_size: Number of papers to fetch per category
        months_to_analyze: How many months of papers to analyze
        include_citations: Whether to include citation data (requires semanticscholar)
    
    Returns:
        Dictionary with trending topics, search queries, and paper samples
    """
    # Import get_arxiv_papers here to avoid circular import
    from arxiv import get_arxiv_papers
    
    # Default to major CS and AI categories if none provided
    if categories is None:
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "quant-ph", "cs.IR"]
    
    logger.info(f"Analyzing trends across {len(categories)} categories")
    analyzer = TrendAnalyzer()
    all_papers = []
    
    # Fetch papers for each category
    for category in categories:
        try:
            logger.info(f"Fetching {initial_sample_size} papers from {category}")
            papers = get_arxiv_papers(
                query=f"cat:{category} AND submittedDate:[{datetime.now().year-1} TO {datetime.now().year}]",
                category=category,
                max_results=initial_sample_size,
                batch_size=100,
                pause_duration=30  # Be gentle with the API
            )
            all_papers.extend(papers)
            logger.info(f"Retrieved {len(papers)} papers from {category}")
        except Exception as e:
            logger.error(f"Failed to fetch papers for {category}: {e}")
    
    # Analyze trends
    logger.info(f"Analyzing trends in {len(all_papers)} papers")
    trending_topics = analyzer.analyze_papers_by_time(all_papers, months_back=months_to_analyze)
    
    # Generate search queries
    search_queries = analyzer.generate_search_queries(trending_topics, top_n=10)
    
    # Add citation analysis if requested
    if include_citations:
        try:
            citation_data = analyze_citations(trending_topics[:20], all_papers)
            for topic in trending_topics[:20]:
                if topic['term'] in citation_data:
                    topic['citation_velocity'] = citation_data[topic['term']]
        except ImportError:
            logger.warning("semanticscholar package not installed. Skipping citation analysis.")
            logger.info("To include citation data, install using: pip install semanticscholar")
    
    # Sample papers for each top trend
    paper_samples = {}
    for trend in trending_topics[:10]:
        term = trend['term']
        matching_papers = []
        for paper in all_papers:
            title = paper.get('title', '').lower()
            summary = paper.get('summary', '').lower()
            if term in title or term in summary:
                matching_papers.append({
                    'title': paper.get('title', 'Untitled'),
                    'authors': paper.get('authors', [])[:3],  # First 3 authors
                    'published': paper.get('published'),
                    'link': paper.get('pdf', paper.get('link', ''))
                })
                if len(matching_papers) >= 3:  # Limit to 3 examples per trend
                    break
        paper_samples[term] = matching_papers
    
    # Format results
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_papers_analyzed": len(all_papers),
        "categories_analyzed": categories,
        "top_trends": trending_topics[:20],
        "search_queries": search_queries,
        "paper_samples": paper_samples
    }
    
    return results

def analyze_citations(trending_topics, papers):
    """
    Analyze citation patterns for trending topics.
    Requires the semanticscholar package.
    
    Returns a dict mapping topic terms to citation velocity scores.
    """
    try:
        from semanticscholar import SemanticScholar
        
        citation_data = {}
        sch = SemanticScholar(timeout=30)  # 30 second timeout
        
        # Find papers matching each trending topic
        for topic in trending_topics:
            term = topic['term']
            matching_papers = []
            
            # Find up to 5 papers that match this topic
            for paper in papers:
                title = paper.get('title', '').lower()
                if term in title and len(matching_papers) < 5:
                    matching_papers.append(paper)
            
            if not matching_papers:
                continue
                
            total_citations = 0
            citation_velocity = 0
            papers_analyzed = 0
            
            for paper in matching_papers:
                try:
                    # Query Semantic Scholar for citation data
                    result = sch.search_paper(paper.get('title', ''))
                    if result and result.get('citationCount'):
                        citations = result.get('citationCount', 0)
                        pub_date = paper.get('published')
                        
                        # Calculate velocity as citations per month
                        if pub_date and isinstance(pub_date, datetime):
                            months_since_pub = max(1, (datetime.now() - pub_date).days / 30)
                            velocity = citations / months_since_pub
                            citation_velocity += velocity
                            total_citations += citations
                            papers_analyzed += 1
                except Exception as e:
                    logger.warning(f"Failed to get citation data: {e}")
            
            # Calculate average velocity
            if papers_analyzed > 0:
                citation_data[term] = {
                    "avg_velocity": citation_velocity / papers_analyzed,
                    "total_citations": total_citations,
                    "papers_analyzed": papers_analyzed
                }
                
        return citation_data
        
    except ImportError:
        logger.warning("semanticscholar package not found")
        return {}

# Add these helper functions to integrate with the TopicProcessor class

def convert_trends_to_topics(trend_results, min_score=5.0):
    """
    Convert trend analysis results into topics compatible with TopicProcessor.
    
    Args:
        trend_results: Output from identify_emerging_trends()
        min_score: Minimum score threshold for including trends
    
    Returns:
        Dictionary of topics in the format expected by TopicProcessor
    """
    from Scrapers.topic_processor import TopicProcessor
    processor = TopicProcessor()
    topics = {}
    
    for i, trend in enumerate(trend_results.get('top_trends', [])):
        if trend.get('score', 0) < min_score:
            continue
            
        # Extract terms and create a topic name
        term = trend['term']
        clean_name = processor.sanitize_topic_name(term)
        topic_key = f"emerging_trend_{i}_{clean_name}"
        
        # Build related terms
        related_terms = term.split()
        for other_trend in trend_results.get('top_trends', []):
            if other_trend['term'] != term and term in other_trend['term']:
                related_terms.extend(other_trend['term'].split())
        
        # Create a unique list of terms
        unique_terms = list(set([t for t in related_terms if len(t) > 2]))
        
        # Get a matching query
        matching_query = next((q['query'] for q in trend_results.get('search_queries', []) 
                              if q['term'] == term), None)
        
        # Predict category
        category = processor.predict_arxiv_category(term, unique_terms)
        
        topics[topic_key] = {
            "query": matching_query or processor.build_optimized_query(unique_terms),
            "category": category,
            "description": f"Emerging Trend: {term.title()}",
            "original_terms": unique_terms,
            "confidence": min(0.9, trend.get('growth', 0) / 10),
            "growth_score": trend.get('score', 0),
            "is_emerging_trend": True
        }
    
    return topics

def get_emerging_topics(categories=None, sample_size=1000):
    """
    Convenience function to get emerging topics in the format expected by 
    scrape_trending_topics().
    
    Args:
        categories: List of arXiv categories to analyze
        sample_size: Number of papers to analyze per category
    
    Returns:
        Dictionary of topics ready for use with scrape_trending_topics()
    """
    # Identify emerging trends
    trend_results = identify_emerging_trends(
        categories=categories,
        initial_sample_size=sample_size,
        include_citations=False  # Set to True if semanticscholar is installed
    )
    
    # Convert to topics
    return convert_trends_to_topics(trend_results)

def load_valid_topics(data_dir=None):
    """
    Load valid topics from trend reports or cache.
    
    Args:
        data_dir: Directory to look for trend reports
    
    Returns:
        Dictionary of valid topics
    """
    from pathlib import Path
    import json
    import os
    
    # Use default path if none provided
    if data_dir is None:
        data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "trend_cache"
    else:
        data_dir = Path(data_dir)
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for trend reports
    logger.info(f"Looking for trend reports in {data_dir}")
    trend_files = list(data_dir.glob("trend_report_*.txt"))
    
    if not trend_files:
        logger.warning("No trend reports found")
        return {}
    
    # Get most recent report
    latest_file = max(trend_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading trends from {latest_file.name}")
    
    try:
        # Extract trending topics from the report
        topics = extract_trending_topics_from_report(latest_file)
        logger.info(f"Loaded {len(topics)} topics from report")
        return topics
    except Exception as e:
        logger.error(f"Failed to load topics: {e}")
        return {}

def extract_trending_topics_from_report(report_path):
    """
    Extract trending topics from a trend report file.
    
    Args:
        report_path: Path to the trend report file
    
    Returns:
        Dictionary of topics
    """
    import json
    from Scrapers.topic_processor import TopicProcessor
    
    processor = TopicProcessor()
    topics = {}
    
    try:
        with open(report_path, 'r') as f:
            trend_data = json.load(f)
        
        for item in trend_data:
            # Create a clean key for this topic
            title = item.get("refined_title", "Untitled Topic")
            clean_name = processor.sanitize_topic_name(title)
            topic_key = f"topic_{item.get('original_id', 'unknown')}_{clean_name}"
            
            # Get related terms
            terms = item.get("top_terms", [])
            
            # Predict the most likely arXiv category
            category = processor.predict_arxiv_category(title, terms)
            
            # Build an optimized query
            query = processor.build_optimized_query(terms)
            
            # Store the topic data
            topics[topic_key] = {
                "query": query,
                "category": category,
                "description": title,
                "original_terms": terms,
                "confidence": item.get("confidence_score", 0.7)
            }
            
        return topics
    except Exception as e:
        logger.error(f"Failed to extract topics from {report_path}: {e}")
        return {}

def fetch_topic_papers(topic_name, topic_info, max_results=50, batch_size=5):
    """
    Fetch papers for a specific topic using optimized query.
    
    Args:
        topic_name: Key identifier for the topic
        topic_info: Dictionary with topic details (query, category)
        max_results: Maximum number of results to return
        batch_size: Number of results per API request
    
    Returns:
        List of papers with added metadata
    """
    from arxiv import get_arxiv_papers
    
    query = topic_info.get("query")
    category = topic_info.get("category")
    
    logger.info(f"Fetching papers for topic: {topic_info.get('description', topic_name)}")
    logger.debug(f"Using query: {query}")
    
    try:
        papers = get_arxiv_papers(
            query=query, 
            category=category,
            max_results=max_results,
            batch_size=batch_size
        )
        
        # Add topic metadata to each paper
        for paper in papers:
            if "metadata" not in paper:
                paper["metadata"] = {}
            paper["metadata"]["topic"] = topic_name
            paper["metadata"]["topic_description"] = topic_info.get("description", "")
            paper["metadata"]["topic_confidence"] = topic_info.get("confidence", 0.0)
            
        logger.info(f"Found {len(papers)} papers for topic {topic_name}")
        return papers
    except Exception as e:
        logger.error(f"Error fetching papers for topic {topic_name}: {e}")
        return []

def scrape_trending_topics(
    max_results_per_topic: int = 100,
    batch_size: int = 10,
    max_parallel_requests: int = 3,
    custom_topics: Dict = None
) -> Dict:
    """
    Scrape papers for trending topics with parallel processing.
    
    Args:
        max_results_per_topic: Maximum number of results per topic
        batch_size: Number of results per API request
        max_parallel_requests: Maximum number of parallel requests
        custom_topics: Optional dictionary of custom topics to use
    
    Returns:
        Dictionary mapping topic keys to lists of papers
    """
    import concurrent.futures
    from topic_processor import TopicProcessor
    
    # Use custom topics if provided, otherwise load from reports
    if custom_topics:
        trending_topics = custom_topics
        logger.info(f"Using {len(trending_topics)} custom topics")
    else:
        # Get trending topics from reports
        trending_topics = load_valid_topics()
        
        # Use fallback topics if none found
        if not trending_topics:
            logger.warning("No valid topics found, using fallback")
            processor = TopicProcessor()
            trending_topics = processor.get_fallback_topics()
    
    all_papers = {}
    logger.info(f"Scraping papers for {len(trending_topics)} trending topics")
    
    # Process topics in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        # Submit tasks
        future_to_topic = {
            executor.submit(fetch_topic_papers, topic_name, topic_info, max_results_per_topic, batch_size): 
            topic_name
            for topic_name, topic_info in trending_topics.items()
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_topic):
            topic_name = future_to_topic[future]
            try:
                papers = future.result()
                all_papers[topic_name] = papers
                logger.info(f"Completed scraping for {topic_name}: {len(papers)} papers")
            except Exception as e:
                logger.error(f"Error processing topic {topic_name}: {e}")
    
    logger.info(f"Completed scraping {sum(len(p) for p in all_papers.values())} papers across {len(all_papers)} topics")
    return all_papers

def analyze_topic_coverage(papers_by_topic):
    """
    Analyze the coverage of papers across topics.
    
    Args:
        papers_by_topic: Dictionary mapping topic keys to lists of papers
    
    Returns:
        Dictionary with coverage statistics
    """
    from collections import Counter
    
    coverage = {}
    
    for topic, papers in papers_by_topic.items():
        if not papers:
            continue
            
        # Get topic description
        description = papers[0].get("metadata", {}).get("topic_description", topic)
        
        # Analyze year distribution
        years = Counter()
        for paper in papers:
            pub_date = paper.get("published")
            if pub_date and hasattr(pub_date, "year"):
                years[pub_date.year] += 1
        
        # Extract sample titles
        sample_titles = [p.get("title", "Untitled") for p in papers[:5]]
        
        # Store coverage data
        coverage[topic] = {
            "total": len(papers),
            "description": description,
            "by_year": dict(years),
            "sample_titles": sample_titles
        }
    
    return coverage
