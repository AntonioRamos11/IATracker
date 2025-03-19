#!/usr/bin/env python3
# filepath: /home/p0wden/Documents/IAResearchAgregator/ProcessingPipiline/ExtractHighlights.py

import os
import json
import logging
import requests
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
nltk.download('punkt')
nltk.download('punkt_tab')
# Download nltk data if needed (first run only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HighlightExtractionError(Exception):
    """Custom exception for highlight extraction errors"""
    pass

def format_paper_for_api(json_path):
    """Format preprocessed PDF data for the highlights API"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        full_text = paper_data.get('text', '')
        metadata = paper_data.get('metadata', {})
        
        # Split text into sections based on common section headers
        # This is a simple heuristic - could be improved with better section detection
        sections = {}
        
        # Try to identify common paper sections
        section_markers = [
            "ABSTRACT", "INTRODUCTION", "BACKGROUND", "RELATED WORK", 
            "METHODOLOGY", "METHOD", "APPROACH", "EXPERIMENTS", "EVALUATION",
            "RESULTS", "DISCUSSION", "CONCLUSION", "REFERENCES"
        ]
        
        current_section = "unknown"
        section_text = []
        
        # Simple section extraction - can be improved
        for line in full_text.split('\n'):
            line_upper = line.strip().upper()
            
            # Check if this line starts a new section
            is_new_section = False
            for marker in section_markers:
                if line_upper.startswith(marker) or line_upper == marker:
                    # Save previous section
                    if section_text:
                        sections[current_section] = '\n'.join(section_text)
                        section_text = []
                    
                    current_section = line_upper.lower()
                    is_new_section = True
                    break
            
            if not is_new_section:
                section_text.append(line)
        
        # Add the last section
        if section_text:
            sections[current_section] = '\n'.join(section_text)
        
        # If we couldn't identify sections, use the full text
        if not sections or len(sections) <= 1:
            sections = {"full_text": full_text}
        
        # Extract abstract if present
        abstract = ""
        if "abstract" in sections:
            abstract = sections["abstract"]
        elif full_text:
            # Heuristic: first paragraph might be abstract
            paragraphs = full_text.split("\n\n")
            if paragraphs:
                abstract = paragraphs[0][:500]  # Limit to first 500 chars
        
        # Format data for API
        api_data = {
            "title": metadata.get('title', 'Untitled Paper'),
            "abstract": abstract,
            "content": full_text,
            "sections": sections,
            "arxiv_id": metadata.get('arxiv_id', ''),
            "authors": metadata.get('author', 'Unknown').split(','),
            "published_date": metadata.get('published', ''),
            "content_hash": metadata.get('content_hash', ''),
            "output_format": "highlights",  # Request highlights
            "max_tokens": 1024  # Reasonable limit
        }
        
        return api_data
    
    except Exception as e:
        logger.error(f"Error formatting paper {json_path}: {str(e)}")
        raise HighlightExtractionError(f"Failed to format paper: {str(e)}")

def preprocess_for_llm(text, max_tokens=10000):
    """
    Preprocess text to fit within token limits using smart extractive techniques.
    Returns a shortened version while preserving key content.
    """
    # Initialize tokenizer for counting tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Count tokens in original text
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    if token_count <= max_tokens:
        return text  # No need to shorten
        
    logger.info(f"Document has {token_count} tokens, reducing to fit {max_tokens} token limit")
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Calculate importance based on simple heuristics
    important_sentences = []
    for i, sentence in enumerate(sentences):
        importance = 0
        # Sentences at beginning and end are often more important
        if i < len(sentences) * 0.1:  # First 10% of sentences
            importance += 3
        elif i > len(sentences) * 0.9:  # Last 10% of sentences
            importance += 2
            
        # Sentences with keywords are important
        keywords = ["propose", "novel", "introduce", "results", "conclude", 
                   "performance", "study", "method", "approach", "contribution"]
        for keyword in keywords:
            if keyword in sentence.lower():
                importance += 1
                
        # Shorter sentences often contain key information
        if len(sentence.split()) < 20:
            importance += 1
            
        important_sentences.append((sentence, importance))
    
    # Sort sentences by importance
    important_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Take most important sentences up to token limit
    selected_sentences = []
    current_tokens = 0
    
    for sentence, _ in important_sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_tokens + sentence_tokens <= max_tokens:
            selected_sentences.append(sentence)
            current_tokens += sentence_tokens
        else:
            break
    
    # Reorder sentences to maintain original flow
    original_order = [(i, s) for i, s in enumerate(sentences) if s in selected_sentences]
    original_order.sort(key=lambda x: x[0])
    
    # Join sentences
    shortened_text = ' '.join([s for _, s in original_order])
    
    # Log reduction stats
    new_token_count = len(tokenizer.encode(shortened_text))
    reduction = 100 - (new_token_count / token_count * 100)
    logger.info(f"Reduced text by {reduction:.1f}% ({token_count} → {new_token_count} tokens)")
    
    return shortened_text

def extract_highlights(json_path, api_url):
    """Extract highlights from a preprocessed paper using the API"""
    try:
        # Format paper data for API
        api_data = format_paper_for_api(json_path)
        
        # Log request details for debugging
        logger.info(f"Sending request for paper: {api_data.get('title', 'Unknown')} ({json_path.name})")
        
        # Preprocess full content to reduce size
        original_content = api_data.get('content', '')
        api_data['content'] = preprocess_for_llm(original_content, max_tokens=8000)
        
        # Preprocess sections as well
        for section_name, section_text in api_data.get('sections', {}).items():
            # Only process large sections
            if len(section_text) > 20000:
                section_limit = 3000 if section_name.lower() in ['abstract', 'introduction', 'conclusion'] else 1500
                api_data['sections'][section_name] = preprocess_for_llm(section_text, max_tokens=section_limit)
        
        # Use a shorter timeout to avoid hanging
        timeout = 60
        
        # Reset CUDA memory between requests (add import gc at top)
        import gc
        gc.collect()
        
        # Send request to API
        response = requests.post(
            api_url,
            json=api_data,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        # More detailed error logging
        if response.status_code != 200:
            logger.error(f"API error {response.status_code}: {response.text[:500]}")
            return False
        
        # Parse response
        result = response.json()
        
        # Store highlights
        highlights_dir = Path("./Database/paper_highlights/")
        highlights_dir.mkdir(parents=True, exist_ok=True)
        
        content_hash = api_data.get('content_hash', '')
        if not content_hash:
            content_hash = json_path.stem
            
        highlight_path = highlights_dir / f"{content_hash}_highlights.json"
        
        with open(highlight_path, 'w', encoding='utf-8') as f:
            json.dump({
                "paper_title": api_data.get('title'),
                "paper_id": content_hash,
                "highlights": result.get('highlights', []),
                "summary": result.get('summary', ''),
                "extraction_date": result.get('timestamp', ''),
                "source_file": str(json_path)
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Extracted highlights for '{api_data.get('title')}' -> {highlight_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"API request failed for {json_path}: {str(e)}")
        
        # Additional debugging for connection issues
        if "Connection refused" in str(e):
            logger.error("The API server doesn't appear to be running. Make sure it's started at the specified URL.")
        elif "timeout" in str(e).lower():
            logger.error("The request timed out. The document may be too large or the server is overloaded.")
        
        return False
    except Exception as e:
        logger.error(f"Failed to extract highlights for {json_path}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())  # More detailed error info
        return False

def process_batch(processed_dir, api_url, max_papers=None, workers=4):
    """Process a batch of papers to extract highlights"""
    json_files = list(processed_dir.glob("*.json"))
    
    if max_papers:
        json_files = json_files[:max_papers]
    
    logger.info(f"Found {len(json_files)} processed papers. Extracting highlights...")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_highlights, f, api_url): f for f in json_files}
        
        # Use tqdm to show progress
        for future in tqdm(futures, desc="Extracting highlights"):
            file_path = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                fail_count += 1
    
    logger.info("=" * 50)
    logger.info(f"Highlight Extraction Summary:")
    logger.info(f"  Total papers:       {len(json_files)}")
    logger.info(f"  Successful:         {success_count}")
    logger.info(f"  Failed:             {fail_count}")
    logger.info("=" * 50)

def process_batch_sequential(processed_dir, api_url, max_papers=None, delay_seconds=5):
    """Process papers sequentially to avoid overwhelming the GPU"""
    json_files = list(processed_dir.glob("*.json"))
    
    if max_papers:
        json_files = json_files[:max_papers]
    
    total_files = len(json_files)
    logger.info(f"Found {total_files} processed papers. Extracting highlights sequentially...")
    
    success_count = 0
    fail_count = 0
    
    # Process files one by one with progress bar
    for idx, json_path in enumerate(tqdm(json_files, desc="Processing papers")):
        try:
            # Process this paper
            logger.info(f"Processing paper {idx+1}/{total_files}: {json_path.name}")
            result = extract_highlights(json_path, api_url)
            
            if result:
                success_count += 1
                logger.info(f"✓ Successfully processed paper {idx+1}/{total_files}")
            else:
                fail_count += 1
                logger.error(f"✗ Failed to process paper {idx+1}/{total_files}")
            
            # Wait between papers to let GPU memory clear
            if idx < total_files - 1 and delay_seconds > 0:
                logger.info(f"Waiting {delay_seconds} seconds before next paper...")
                time.sleep(delay_seconds)
                
        except Exception as e:
            logger.error(f"Error processing {json_path}: {str(e)}")
            fail_count += 1
    
    logger.info("=" * 50)
    logger.info(f"Highlight Extraction Summary:")
    logger.info(f"  Total papers:       {total_files}")
    logger.info(f"  Successful:         {success_count}")
    logger.info(f"  Failed:             {fail_count}")
    logger.info("=" * 50)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Extract paper highlights using an API")
    parser.add_argument("--api-url", default="http://localhost:5000/api/process_paper",
                        help="API endpoint URL")
    parser.add_argument("--processed-dir", default="./Database/processed_pdfs/",
                        help="Directory containing processed paper JSONs")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Maximum number of papers to process (None for all)")
    parser.add_argument("--parallel", action="store_true",
                        help="Process papers in parallel (default: sequential)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (only with --parallel)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Delay in seconds between papers (only in sequential mode)")
    parser.add_argument("--test", action="store_true",
                        help="Run a simple API test with minimal data")
    
    args = parser.parse_args()
    
    if args.test:
        test_api_connection(args.api_url)
        return
    
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory {processed_dir} does not exist")
    
    # Add import for time.sleep
    import time
    
    if args.parallel:
        logger.info("Running in parallel mode with multiple workers")
        process_batch(processed_dir, args.api_url, args.max_papers, args.workers)
    else:
        logger.info(f"Running in sequential mode with {args.delay}s delay between papers")
        process_batch_sequential(processed_dir, args.api_url, args.max_papers, args.delay)

def test_api_connection(api_url):
    """Test the API with a minimal request to verify it's working"""
    logger.info(f"Testing API connection to: {api_url}")
    
    # Create a minimal test payload
    test_data = {
        "title": "Test Paper",
        "abstract": "This is a test abstract for API connection testing.",
        "content": "Short content for testing the API connection.",
        "output_format": "highlights",
        "max_tokens": 100
    }
    
    try:
        logger.info("Sending test request...")
        response = requests.post(
            api_url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        logger.info(f"API response status: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"Success! Response: {response.text[:200]}...")
        else:
            logger.error(f"API error: {response.text}")
            
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise SystemExit(1) from e