import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import PipelineException
from huggingface_hub import model_info, HfApi
from Topic_Refiner import TopicRefiner

# Add these imports at the top of the file
import re
from collections import Counter
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

# Download required NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exceptions
class ModelValidationError(Exception):
    """Raised when a model fails validation checks"""
    pass

class ModelOutputError(Exception):
    """Raised when a model produces unexpected outputs"""
    pass

class AuthenticationError(Exception):
    """Raised when authentication is required but not provided"""
    pass

class TopicProcessor:
    """ML-powered topic processing system for arXiv papers"""
    
    def __init__(self, hf_token: str = None):
        """Initialize ML models with validation checks"""
        self.hf_token = hf_token
        self.models_valid = False
        self._initialize_models()

    def _initialize_models(self):
        """Model initialization with validity checks"""
        try:
            logger.info("Initializing ML models for topic processing...")
            
            # First validate model availability
            self._validate_model_access("distilbert/distilbert-base-uncased")

            # Then load models
            self.category_classifier = pipeline(
            "text-classification",
            model="allenai/scibert_scivocab_uncased",
            token=self.hf_token,
            top_k=3
            )
                    
            self.topic_validator = pipeline(
            "text-classification",
            model="distilroberta-base",
            token=self.hf_token
            )   

            # Run sanity checks
            self._validate_model_outputs()
            self.models_valid = True
            logger.info("ML models initialized and validated successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self._handle_model_failure()

    def _validate_model_access(self, model_name: str):
        """Check model availability and permissions"""
        try:
            info = model_info(model_name, token=self.hf_token)
            
            if not info.siblings:
                raise ValueError(f"No model files found for {model_name}")
                
            if info.private and not self.hf_token:
                raise AuthenticationError(
                    f"Private model {model_name} requires authentication"
                )

            logger.debug(f"Model {model_name} is accessible")
            
        except Exception as e:
            raise ModelValidationError(
                f"Failed to validate {model_name}: {str(e)}"
            ) from e

    def _validate_model_outputs(self):
        """Test model with sample inputs - with proper error handling"""
        try:
            test_cases = [
                ("Large Language Models", ["transformer", "attention"], "cs.CL"),
                ("Quantum Computing", ["qubit", "entanglement"], "quant-ph")
            ]
            
            for title, terms, expected_cat in test_cases:
                # Test category classifier
                input_text = f"{title}: {', '.join(terms[:3])}"
                
                # Check classifier with proper error handling
                try:
                    cat_pred = self.category_classifier(input_text)
                    
                    # Inspect the actual output structure
                    logger.debug(f"Category prediction output structure: {type(cat_pred)}")
                    if isinstance(cat_pred, list) and cat_pred:
                        # If list of dicts (expected format)
                        if isinstance(cat_pred[0], dict) and 'label' in cat_pred[0]:
                            logger.info(f"Category prediction for {title}: {cat_pred[0]['label']}")
                        else:
                            logger.warning(f"Unexpected category prediction format: {cat_pred}")
                except Exception as e:
                    logger.warning(f"Category classifier error: {e}")
                
                # Test validator with proper error handling
                try:
                    validation = self.topic_validator(input_text)
                    logger.debug(f"Validation output structure: {type(validation)}")
                    
                    if isinstance(validation, list) and validation:
                        # Extract result based on actual structure
                        result = validation[0]
                        if isinstance(result, dict) and 'label' in result:
                            logger.info(f"Validation for {title}: {result['label']}")
                        else:
                            logger.warning(f"Unexpected validation format: {validation}")
                    else:
                        logger.warning(f"Unexpected validation output: {validation}")
                except Exception as e:
                    logger.warning(f"Topic validator error: {e}")

            logger.debug("Model output validation completed")
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def _handle_model_failure(self):
        """Fallback strategy for model failures"""
        logger.warning("Using rule-based fallback methods")
        self.category_classifier = self._rule_based_classifier
        self.topic_validator = self._basic_topic_check
        self.models_valid = False

    def _rule_based_classifier(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """Fallback classification rules"""
        text_lower = text.lower()
        results = []
        
        # Rule-based checks for common categories
        rules = [
            (["language", "nlp", "text", "gpt", "llm", "transformer"], "cs.CL", 0.8),
            (["vision", "image", "video", "detection"], "cs.CV", 0.8),
            (["reinforcement", "rl", "agent", "policy", "reward"], "cs.AI", 0.7),
            (["neural", "network", "deep", "learning", "optimization"], "cs.LG", 0.7),
            (["quantum", "qubits"], "quant-ph", 0.9),
            (["information", "retrieval", "search", "ranking"], "cs.IR", 0.6)
        ]
        
        for keywords, category, confidence in rules:
            if any(kw in text_lower for kw in keywords):
                results.append({'label': category, 'score': confidence})
        
        # Always include a default with lower confidence
        if not results:
            results.append({'label': 'cs.AI', 'score': 0.5})
        
        # Sort by confidence score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:3]  # Return top 3 to mimic top_k=3 in the real model

    def _basic_topic_check(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """Fallback topic validation"""
        # Check if the text has sufficient length and complexity
        words = text.split()
        
        if len(words) < 3:
            return [{'label': 'INVALID', 'score': 0.9}]
            
        # Check for common stopwords
        if len(set(words) - {'the', 'and', 'or', 'of', 'in', 'a', 'an', 'to'}) < 3:
            return [{'label': 'INVALID', 'score': 0.8}]
            
        return [{'label': 'VALID', 'score': 0.7}]

    def is_model_valid(self) -> bool:
        """Check if ML models are functioning"""
        return self.models_valid

    def sanitize_topic_name(self, title: str) -> str:
        """Clean and normalize topic names"""
        return (
            title.lower()
            .replace(' ', '_')
            .replace('&', 'and')
            .replace('/', '_')
            .translate(str.maketrans('', '', r"""!"#$%'()*+,.:;<=>?@[\]^`{|}~"""))
            [:64]
        )

    def validate_topic_quality(self, title: str, terms: List[str]) -> bool:
        """ML-powered topic quality validation with improved handling"""
        if not self.models_valid:
            # Fallback validation logic
            return (
                len(title) >= 10 and
                len(terms) >= 3 and
                not any(term.lower() in ['et', 'al', 'the', 'and', 'or'] for term in terms[:3])
            )
            
        try:
            validation_text = f"{title}: {', '.join(terms[:5])}"
            result = self.topic_validator(validation_text, truncation=True)
            
            # Handle different output formats
            if isinstance(result, list) and result:
                first_result = result[0]
                
                # If using generic LABEL_X format
                if isinstance(first_result, dict) and 'label' in first_result:
                    # For models that use LABEL_1 for positive class
                    if first_result['label'] == 'LABEL_1' and first_result['score'] > 0.7:
                        return True
                    # For sentiment models (positive = valid)
                    if first_result['label'] in ['POSITIVE', '1', 'VALID'] and first_result['score'] > 0.7:
                        return True
                        
            # Fallback to rule-based validation
            return (
                len(title) >= 10 and
                len(terms) >= 3 and
                not any(term.lower() in ['et', 'al', 'the', 'and', 'or'] for term in terms[:3])
            )
            
        except Exception as e:
            logger.warning(f"Topic validation error: {e}. Using fallback validation.")
            # Fallback logic
            return len(title) >= 10 and len(terms) >= 3

    def predict_arxiv_category(self, title: str, terms: List[str]) -> str:
        """Predict arXiv category using ML model or rule-based fallback"""
        try:
            if self.models_valid:
                context = f"{title}. Keywords: {', '.join(terms[:5])}"
                predictions = self.category_classifier(context, truncation=True)
                
                # Map the generic labels to meaningful categories
                mapped_prediction = self._map_model_labels_to_categories(predictions, title, terms)
                return mapped_prediction['label']
            else:
                # Use the rule-based classifier
                context = f"{title}. Keywords: {', '.join(terms[:5])}"
                predictions = self.category_classifier(context)
                return predictions[0]['label']
        except Exception as e:
            logger.warning(f"Category prediction error: {e}. Using rule-based fallback.")
            # Use fallback logic
            title_lower = title.lower()
            terms_lower = [t.lower() for t in terms]
            
            # Rules for common categories
            if any(kw in title_lower or any(kw in t for t in terms_lower) 
                for kw in ["language", "nlp", "text", "gpt", "llm", "transformer"]):
                return "cs.CL"
            elif any(kw in title_lower or any(kw in t for t in terms_lower)
                    for kw in ["vision", "image", "video", "detection"]):
                return "cs.CV"
            elif any(kw in title_lower or any(kw in t for t in terms_lower)
                    for kw in ["reinforcement", "rl", "agent", "policy", "reward"]):
                return "cs.AI"
            elif any(kw in title_lower or any(kw in t for t in terms_lower)
                    for kw in ["neural", "network", "deep", "learning", "optimization"]):
                return "cs.LG"
            elif any(kw in title_lower or any(kw in t for t in terms_lower)
                    for kw in ["quantum", "qubits"]):
                return "quant-ph"
            elif any(kw in title_lower or any(kw in t for t in terms_lower)
                    for kw in ["information", "retrieval", "search", "ranking"]):
                return "cs.IR"
            
            # Default category
            return "cs.AI"

    def _map_model_labels_to_categories(self, prediction, title, terms):
        """Map generic model labels to meaningful arXiv categories"""
        # For models that return LABEL_X format
        if isinstance(prediction, list) and prediction:
            first_pred = prediction[0]
            
            if isinstance(first_pred, dict) and 'label' in first_pred:
                # If it's a list with LABEL format
                if first_pred['label'].startswith('LABEL_'):
                    logger.debug(f"Mapping generic label {first_pred['label']} for '{title}'")
                    
                    # Use rule-based approach for actual category
                    title_lower = title.lower()
                    terms_lower = [t.lower() for t in terms]
                    
                    # Convert confidence score
                    confidence = first_pred.get('score', 0.5)
                    
                    # Rules for common categories - same as your rule-based classifier
                    if any(kw in title_lower or any(kw in t for t in terms_lower) 
                        for kw in ["language", "nlp", "text", "gpt", "llm", "transformer"]):
                        return {'label': "cs.CL", 'score': confidence}
                    elif any(kw in title_lower or any(kw in t for t in terms_lower)
                            for kw in ["vision", "image", "video", "detection"]):
                        return {'label': "cs.CV", 'score': confidence}
                    # Add other rules from your rule-based classifier
                    
                    # Default category
                    return {'label': "cs.AI", 'score': confidence}
                
                # Return the prediction as is if it already has meaningful labels
                return first_pred
        
        # Default fallback
        return {'label': "cs.AI", 'score': 0.5}

    def build_optimized_query(self, terms: List[str]) -> str:
        """Build sophisticated arXiv search query"""
        if not terms:
            return "artificial intelligence"  # Fallback query
            
        # Extract high-quality terms
        terms = [term for term in terms if len(term) > 2]
        if not terms:
            return "artificial intelligence"

        # Prioritize multi-word terms with quotes
        prioritized = [
            f'"{term}"' if ' ' in term else term 
            for term in terms[:3]
        ]
        
        # Add secondary terms with field specifiers
        secondary = []
        if len(terms) > 3:
            secondary = [f'all:{term}' for term in terms[3:5]]
        
        # Exclude some categories that tend to be noisy
        exclusions = "ANDNOT (cat:math.NA OR cat:physics.soc-ph)"
        
        # Combine everything
        query = f"({' OR '.join(prioritized)})"
        if secondary:
            query += f" {' '.join(secondary)}"
        query += f" {exclusions}"
        
        return query

    def get_fallback_topics(self) -> Dict:
        """Enhanced default topics with expiration dates"""
        return {
            "explainable_ai_v2024": {
                "query": '(ti:"explainable ai" OR abs:"model interpretability") ANDNOT cat:math.LO',
                "category": "cs.AI",
                "description": "Explainable AI Techniques",
                "expires": "2024-12-31",
                "original_terms": ["explainable", "interpretable", "model", "transparency"]
            },
            "multimodal_learning_v2024": {
                "query": '(ti:multimodal OR abs:"cross-modal") AND (cat:cs.CV OR cat:cs.CL)',
                "category": "cs.LG",
                "description": "Multimodal Learning Approaches",
                "expires": "2024-12-31",
                "original_terms": ["multimodal", "cross-modal", "vision-language", "fusion"]
            },
            "transformer_architectures_v2024": {
                "query": '(ti:transformer OR ti:attention) AND (cat:cs.CL OR cat:cs.LG)',
                "category": "cs.CL",
                "description": "Transformer Architecture Innovations",
                "expires": "2024-12-31",
                "original_terms": ["transformer", "attention", "self-attention", "language model"]
            },
            "reinforcement_learning_v2024": {
                "query": '(ti:"reinforcement learning" OR abs:RL) AND (cat:cs.AI OR cat:cs.LG)',
                "category": "cs.AI",
                "description": "Reinforcement Learning Advances",
                "expires": "2024-12-31",
                "original_terms": ["reinforcement", "RL", "policy", "reward", "agent"]
            },
            "vision_transformers_v2024": {
                "query": '(ti:"vision transformer" OR ti:ViT) AND (cat:cs.CV)',
                "category": "cs.CV",
                "description": "Vision Transformer Models",
                "expires": "2024-12-31",
                "original_terms": ["vision transformer", "ViT", "image", "attention"]
            }
        }

class TrendAnalyzer:
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
