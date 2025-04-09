#!/usr/bin/env python3
import os
import json
import glob
import logging

import argparse
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from TimeMachine import ResearchTimeMachine 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperSearchEngine:
    def __init__(self, papers_dir="./Database/processed_pdfs", highlights_dir="./Database/paper_highlights"):
        self.papers_dir = Path(papers_dir)
        self.highlights_dir = Path(highlights_dir)
        self.papers = {}
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = None
        self.paper_ids = []
        self.model = None
        self.tokenizer = None
        
    def load_papers(self):
        """Load all papers and their highlights from the database"""
        logger.info("Loading papers from database...")
        
        # Load processed papers
        paper_files = list(self.papers_dir.glob("*.json"))
        highlight_files = list(self.highlights_dir.glob("*_highlights.json"))
        
        # Create a mapping of paper IDs to highlight files
        highlight_map = {}
        for hf in highlight_files:
            paper_id = hf.stem.split("_highlights")[0]
            highlight_map[paper_id] = hf
            
        # Load paper content and highlights
        for paper_file in paper_files:
            paper_id = paper_file.stem
            
            try:
                with open(paper_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Initialize paper entry
                self.papers[paper_id] = {
                    "content": paper_data.get("text", ""),
                    "metadata": {
                        "title": paper_data.get("metadata", {}).get("title", paper_id),
                        "authors": paper_data.get("metadata", {}).get("author", "Unknown"),
                        "published": paper_data.get("metadata", {}).get("published", ""),
                        "arxiv_id": paper_data.get("metadata", {}).get("arxiv_id", "")
                    },
                    "highlights": {}
                }
                
                # Add highlights if available
                if paper_id in highlight_map:
                    with open(highlight_map[paper_id], 'r', encoding='utf-8') as f:
                        highlight_data = json.load(f)
                        self.papers[paper_id]["highlights"] = highlight_data
                        
                # Store paper ID for indexing
                self.paper_ids.append(paper_id)
                
            except Exception as e:
                logger.error(f"Error loading paper {paper_id}: {str(e)}")
        
        logger.info(f"Loaded {len(self.papers)} papers")
        
    def build_index(self):
        """Build search index from paper content"""
        if not self.papers:
            self.load_papers()
            
        logger.info("Building search index...")
        
        # Prepare documents for indexing
        documents = []
        for paper_id in self.paper_ids:
            paper = self.papers[paper_id]
            
            # Combine paper title, content and highlights
            doc_text = f"{paper['metadata']['title']} {paper['content']}"
            
            # Add highlights if available
            if paper["highlights"]:
                highlights = paper["highlights"]
                if isinstance(highlights, dict):
                    # Extract different types of highlights
                    for section, content in highlights.items():
                        if isinstance(content, list):
                            doc_text += " " + " ".join(content)
                        elif isinstance(content, str):
                            doc_text += " " + content
                        elif isinstance(content, dict):
                            for k, v in content.items():
                                if isinstance(v, str):
                                    doc_text += " " + v
                                elif isinstance(v, list):
                                    doc_text += " " + " ".join(v)
            
            documents.append(doc_text)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        logger.info(f"Built index with {self.tfidf_matrix.shape[1]} features")
        
    def search(self, query, top_k=5):
        """Search papers based on query and return top_k results"""
        if self.tfidf_matrix is None:
            self.build_index()
            
        # Transform query to TF-IDF
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top_k papers
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            paper_id = self.paper_ids[idx]
            paper = self.papers[paper_id]
            
            results.append({
                "paper_id": paper_id,
                "title": paper["metadata"]["title"],
                "authors": paper["metadata"]["authors"],
                "published": paper["metadata"]["published"],
                "score": float(similarity_scores[idx]),
                "highlights": paper["highlights"] if "highlights" in paper else {}
            })
            
        return results
    
    def initialize_model(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        """Initialize LLM for answer generation"""
        logger.info(f"Loading model: {model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_answer(self, query, search_results):
        """Generate answer to query based on retrieved papers"""
        if self.model is None:
            self.initialize_model()
            
        # Prepare context from search results
        context = ""
        for i, result in enumerate(search_results):
            context += f"\nPaper {i+1}: {result['title']}\n"
            context += f"Authors: {result['authors']}\n"
            
            # Add highlights
            highlights = result.get("highlights", {})
            if "contribution" in highlights:
                context += f"Main contribution: {highlights['contribution']}\n"
            
            if "methods" in highlights and isinstance(highlights["methods"], list):
                context += "Methods:\n" + "\n".join([f"- {m}" for m in highlights["methods"]]) + "\n"
                
            if "results" in highlights and isinstance(highlights["results"], list):
                context += "Results:\n" + "\n".join([f"- {r}" for r in highlights["results"]]) + "\n"
                
            if "impact" in highlights:
                context += f"Impact: {highlights['impact']}\n"
            
            # Add topics if available
            if "topics" in highlights:
                if isinstance(highlights["topics"], list):
                    context += f"Topics: {', '.join(highlights['topics'])}\n"
                        
            # Add more fields as necessary
        
        # Prepare prompt
        prompt = f"""Based on the following papers, answer the question: "{query}"

{context}

Answer:"""

        # Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the answer part (after the prompt)
        answer = answer[len(prompt):]
        
        return answer.strip()

    def analyze_temporal_patterns(self, query):
        """Analyze temporal patterns for the query topic"""
        time_machine = ResearchTimeMachine(
            papers_dir=self.papers_dir, 
            highlights_dir=self.highlights_dir
        )
        time_machine.load_papers()
        
        # Visualize concept evolution
        time_machine.visualize_concept_evolution(query)
        
        # Predict future trends
        trends = time_machine.predict_future_trends(query)
        
        return {
            'evolution_path': f"research_evolution_{query.replace(' ', '_')}.html",
            'trend_forecast': trends
        }

    def generate_research_alternatives(self, query, num_alternatives=3):
        """Generate alternative research paths for the topic"""
        time_machine = ResearchTimeMachine(
            papers_dir=self.papers_dir, 
            highlights_dir=self.highlights_dir
        )
        time_machine.load_papers()
        
        alternatives = time_machine.generate_alternative_paths(query, num_alternatives)
        return alternatives

def main():
    parser = argparse.ArgumentParser(description="Search through research papers and analyze")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--generate_answer", action="store_true", help="Generate answer based on search results")
    
    # Add new timeline analysis arguments
    parser.add_argument("--temporal", action="store_true", help="Analyze temporal patterns")
    parser.add_argument("--forecast", action="store_true", help="Predict future research trends")
    parser.add_argument("--alternative-paths", type=int, help="Generate N alternative research paths")
    
    args = parser.parse_args()
    
    search_engine = PaperSearchEngine()
    
    if args.temporal:
        print(f"Analyzing temporal patterns for: {args.query}")
        results = search_engine.analyze_temporal_patterns(args.query)
        print(f"Evolution visualization saved to: {results['evolution_path']}")
        if results['trend_forecast']:
            print(f"Trend direction: {results['trend_forecast']['trend_direction']}")
            
    if args.forecast:
        print(f"Forecasting research trends for: {args.query}")
        time_machine = ResearchTimeMachine(
            papers_dir=search_engine.papers_dir, 
            highlights_dir=search_engine.highlights_dir
        )
        time_machine.load_papers()
        forecast = time_machine.predict_future_trends(args.query)
        print(f"Forecast visualization saved")
        
    if args.alternative_paths:
        print(f"Generating {args.alternative_paths} alternative research paths for: {args.query}")
        alternatives = search_engine.generate_research_alternatives(args.query, args.alternative_paths)
        for i, alt in enumerate(alternatives):
            print(f"\nAlternative Path {i+1}:")
            print(f"Approach: {alt['approach'][:100]}...")
    
    results = search_engine.search(args.query, top_k=args.top_k)
    
    print(f"=== Search Results for: {args.query} ===\n")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
        print(f"   Authors: {result['authors']}")
        print(f"   Score: {result['score']:.4f}")
        print("")
    
    if args.generate_answer:
        print("Generating answer based on search results...")
        answer = search_engine.generate_answer(args.query, results)
        print("\n=== Answer ===\n")
        print(answer)

if __name__ == "__main__":
    main()