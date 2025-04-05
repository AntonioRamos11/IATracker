#!/usr/bin/env python3
# filepath: /home/p0wden/Documents/IAResearchAgregator/ProcessingPipiline/ExtractHighlights.py
import re
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
from transformers import BertTokenizer, AutoTokenizer
import re
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

# Add this function to parse the structured response text into highlights
def extract_highlights_from_response_text(response_text):
    """Extract structured highlights from response text"""
    highlights = []
    sections = response_text.split("\n\n")
    
    for section in sections:
        # Extract bullet points from the response
        lines = section.split("\n")
        for line in lines:
            # Look for lines starting with bullet points or numbered items
            line = line.strip()
            if line.startswith("- ") or line.startswith("• "):
                # Add the bullet point content as a highlight
                highlight = line[2:].strip()
                if highlight and len(highlight) > 10:  # Skip very short items
                    highlights.append(highlight)
            # Also catch numbered lists with format "1. text" or "1) text"
            elif re.match(r"^\d+[\.\)]", line):
                # This is a numbered item, skip the number
                match = re.match(r"^\d+[\.\)]\s*", line)
                if match:
                    highlight = line[len(match.group()):].strip()
                    if highlight and len(highlight) > 10:
                        highlights.append(highlight)
    
    return highlights

def test_api_format(api_url):
    """Test the API and print the exact format of the response"""
    logger.info(f"Testing API response format from: {api_url}")
    
    # Create a simple test payload
    test_data = {
        "title": "Test Paper for Format Checking",
        "abstract": "This is a test abstract for checking the API response format.",
        "content": "This paper presents a novel method for understanding API response formats.",
        "sections": {"introduction": "APIs have different response formats."},
        "output_format": "highlights",
        "max_tokens": 200
    }
    
    try:
        logger.info("Sending format test request...")
        response = requests.post(
            api_url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Success! Response structure:\n")
            logger.info(f"Response keys: {list(result.keys())}")
            
            # Print the complete structure
            logger.info(f"Full response: {json.dumps(result, indent=2)}")
            
            # Check for highlights and summary specifically
            logger.info(f"Contains 'highlights': {'highlights' in result}")
            logger.info(f"Contains 'summary': {'summary' in result}")
            
            # Look for other possible keys that might contain highlights/summary
            possible_highlight_keys = ['highlights', 'key_points', 'main_points', 'results', 'findings']
            possible_summary_keys = ['summary', 'abstract', 'overview', 'description']
            
            for key in possible_highlight_keys:
                if key in result:
                    logger.info(f"Found potential highlights in key: '{key}'")
            
            for key in possible_summary_keys:
                if key in result:
                    logger.info(f"Found potential summary in key: '{key}'")
                    
        else:
            logger.error(f"API error: {response.text}")
            
    except Exception as e:
        logger.error(f"Format test failed: {str(e)}")

def get_model_capacity(api_url):
    """Get model's actual context window size and model name from the API health endpoint."""
    try:
        base_url = api_url.rsplit('/', 1)[0]
        response = requests.get(f"{base_url}/health", timeout=10)
        data = response.json()
        model_name = data.get('model', 'unknown')
        
        # Context window size database (example limits)
        model_limits = {
            'deepseek-r1-distill-qwen-7b': 32768,
            'llama2-13b': 4096,
            'gpt-4': 8192,
            'default': 4096
        }
        capacity = model_limits.get(model_name.lower(), 4096)
        return capacity, model_name
    except Exception as e:
        logger.error(f"Error getting model capacity: {e}")
        return 4096, 'unknown'

def process_long_text(text, tokenizer, max_tokens, chunk_size=2048, overlap=512):
    """Process long text using sliding window chunking."""
    # Tokenize once and obtain token IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return text  # Within budget, no further processing needed
    
    # Sliding window chunking with a fixed chunk size and overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap tokens to preserve context
        
        if len(chunks) >= 8:  # Safety limit to prevent excessive chunking
            break

    # Decode, clean, and join chunks
    processed_chunks = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        cleaned = clean_text_chunk(chunk_text)
        processed_chunks.append(cleaned)
        
    return "\n\n[CONTINUED] ".join(processed_chunks)

def clean_text_chunk(text):
    """Clean and format text chunks while preserving structure."""
    # Remove excessive newlines (preserve at most two consecutive ones)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove incomplete word fragments at the end of the chunk
    text = re.sub(r'\s[^\s.]{1,20}$', '', text)
    # Fix hyphenated word breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def extract_highlights(json_path, api_url, api_max_tokens=None):
    """Extract highlights using modern tokenization and chunking strategies"""
    try:
        # Get model capacity (and model name) if not provided
        if api_max_tokens is None:
            api_max_tokens, model_name = get_model_capacity(api_url)  # Modified to get model name
        else:
            # If overridden, use default model name for safety
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        
        # Format paper data for API
        api_data = format_paper_for_api(json_path)
        
        # Log request details
        logger.info(f"Processing: {api_data.get('title', 'Unknown')} ({json_path.name})")
        
        # Initialize appropriate tokenizer based on model capacity
        tokenizer = AutoTokenizer.from_pretrained(
            model_name if model_name and model_name != 'unknown' else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            trust_remote_code=True
        )
        
        # Dynamic budget allocation based on model capacity
        metadata_budget = int(api_max_tokens * 0.1)  # 10% for metadata
        content_budget = int(api_max_tokens * 0.6)   # 60% for main content
        sections_budget = api_max_tokens - content_budget - metadata_budget
        
        logger.info(f"Token budget: Total={api_max_tokens}, Content={content_budget}, Sections={sections_budget}")

        # Process main content with sliding-window chunking
        original_content = api_data.get('content', '')
        api_data['content'] = process_long_text(
            text=original_content,
            tokenizer=tokenizer,
            max_tokens=content_budget,
            chunk_size=2048,  # Adjust based on model's optimal chunk size
            overlap=512
        )

        # Process each section with hierarchical chunking
        sections = api_data.get('sections', {})
        for section in sections:
            sections[section] = process_long_text(
                text=sections[section],
                tokenizer=tokenizer,
                max_tokens=sections_budget // max(len(sections), 1),
                chunk_size=1024,
                overlap=256
            )
        api_data['sections'] = sections

        # (Insert your API call here with the processed api_data)
        response = requests.post(
            api_url,
            json=api_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully processed: {api_data.get('title', 'Unknown')}")
            highlight_path = json_path.with_suffix('.highlights.json')
            with open(highlight_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, ensure_ascii=False, indent=2)
            return response.json()
        else:
            logger.error(f"API error {response.status_code}: {response.text}")
            return False

    except requests.RequestException as e:
        logger.error(f"API request failed for {json_path}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Failed to extract highlights for {json_path}: {str(e)}")
        return False

def process_batch(processed_dir, api_url, max_papers=None, workers=4, max_tokens=None):
    """Process a batch of papers to extract highlights"""
    json_files = list(processed_dir.glob("*.json"))
    
    if max_papers:
        json_files = json_files[:max_papers]
    
    logger.info(f"Found {len(json_files)} processed papers. Extracting highlights...")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_highlights, f, api_url, max_tokens): f for f in json_files}
        
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

def process_batch_sequential(processed_dir, api_url, max_papers=None, delay_seconds=5, max_tokens=None):
    """Process papers sequentially with dynamic token limits"""
    # Get model capacity once at the beginning
    api_max_tokens = None
    if max_tokens is None:
        api_max_tokens, _ = get_model_capacity(api_url)
    else:
        api_max_tokens = max_tokens
    
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
            # Process this paper with the determined token limit
            logger.info(f"Processing paper {idx+1}/{total_files}: {json_path.name}")
            result = extract_highlights(json_path, api_url, api_max_tokens)
            
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
    parser.add_argument("--test-api-format", action="store_true",
                        help="Test the API response format")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override the maximum token limit (default: query from API)")
    
    args = parser.parse_args()
    
    if args.test:
        test_api_connection(args.api_url)
        return
    
    if args.test_api_format:
        test_api_format(args.api_url)
        return
    
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory {processed_dir} does not exist")
    
    # Add import for time.sleep
    import time
    
    if args.parallel:
        logger.info("Running in parallel mode with multiple workers")
        process_batch(processed_dir, args.api_url, args.max_papers, args.workers, args.max_tokens)
    else:
        logger.info(f"Running in sequential mode with {args.delay}s delay between papers")
        process_batch_sequential(processed_dir, args.api_url, args.max_papers, args.delay, args.max_tokens)

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