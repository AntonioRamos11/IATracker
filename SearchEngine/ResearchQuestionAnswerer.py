#!/usr/bin/env python3
import json
import re
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchQuestionAnswerer:
    """Class dedicated to answering research questions based on papers highlights"""
    
    def __init__(self, highlights_dir="./Database/paper_highlights", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.highlights_dir = Path(highlights_dir)
        self.papers = {}
        self.paper_ids = []
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = None
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_papers(self):
        """Load paper highlights from the database"""
        logger.info("Loading paper highlights...")
        
        # Load highlight files
        highlight_files = list(self.highlights_dir.glob("*_highlights.json"))
        
        # Process each highlight file
        for hf in highlight_files:
            paper_id = hf.stem.split("_highlights")[0]
            
            try:
                with open(hf, 'r', encoding='utf-8') as f:
                    highlight_data = json.load(f)
                
                # Get metadata
                metadata = highlight_data.get("metadata", {})
                main_staff = highlight_data.get("main_staff", {})
                
                # Extract structured highlights from full_response
                full_response = highlight_data.get("full_response", "")
                
                # Parse structured sections from full_response
                core_innovation = self._extract_section(full_response, "Core Innovation")
                technical_approach = self._extract_section(full_response, "Technical Approach")
                quantitative_findings = self._extract_section(full_response, "Quantitative Findings")
                limitations = self._extract_section(full_response, "Limitations & Assumptions")
                domain_impact = self._extract_section(full_response, "Domain Impact")
                
                # Initialize paper entry with parsed highlights
                self.papers[paper_id] = {
                    "title": metadata.get("title", main_staff.get("title", paper_id)),
                    "authors": metadata.get("author", main_staff.get("authors", ["Unknown"])),
                    "published": metadata.get("published", "Unknown"),
                    "abstract": main_staff.get("abstract", ""),
                    "topics": highlight_data.get("topics", []),
                    "core_innovation": core_innovation,
                    "technical_approach": technical_approach,
                    "quantitative_findings": quantitative_findings,
                    "limitations": limitations,
                    "domain_impact": domain_impact,
                    "full_text": full_response
                }
                
                # Store paper ID for indexing
                self.paper_ids.append(paper_id)
                
            except Exception as e:
                logger.error(f"Error loading highlights for {paper_id}: {str(e)}")
        
        logger.info(f"Loaded highlights for {len(self.papers)} papers")
    
    def _extract_section(self, text, section_name):
        """Extract a specific section from the structured highlights"""
        # Different patterns to try
        patterns = [
            rf"## \d+\.\s*{section_name}\n(.*?)(?=\n## \d+\.|$)",  # Markdown style
            rf"\d+\.\s*{section_name}\n(.*?)(?=\n\d+\.|$)",         # Numbered style
            rf"{section_name}:(.*?)(?=\n[A-Z][\w\s]+:|$)"           # Simple colon style
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                return matches.group(1).strip()
        
        return ""  # Return empty string if section not found
        
    def build_index(self):
        """Build search index from paper highlights"""
        if not self.papers:
            self.load_papers()
            
        logger.info("Building question answering index...")
        
        # Prepare documents for indexing
        documents = []
        for paper_id in self.paper_ids:
            paper = self.papers[paper_id]
            
            # Combine all relevant paper information for indexing
            doc_text = f"{paper['title']} {paper['abstract']} "
            
            # Add core innovation
            if paper["core_innovation"]:
                doc_text += f"{paper['core_innovation']} "
                
            # Add technical approach
            if paper["technical_approach"]:
                doc_text += f"{paper['technical_approach']} "
                
            # Add findings
            if paper["quantitative_findings"]:
                doc_text += f"{paper['quantitative_findings']} "
                
            # Add domain impact
            if paper["domain_impact"]:
                doc_text += f"{paper['domain_impact']} "
                
            # Add topics
            if paper["topics"]:
                doc_text += " ".join(paper["topics"]) + " "
            
            documents.append(doc_text)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        logger.info(f"Built question answering index with {len(documents)} papers")
        
    def find_relevant_papers(self, question, top_k=5):
        """Find papers most relevant to the question"""
        if self.tfidf_matrix is None:
            self.build_index()
            
        # Transform question to TF-IDF
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarity
        similarity_scores = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        
        # Get top_k papers
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        # Return results
        relevant_papers = []
        for idx in top_indices:
            paper_id = self.paper_ids[idx]
            paper = self.papers[paper_id]
            
            # Only include papers with significant relevance
            if similarity_scores[idx] > 0.1:  # Threshold for relevance
                relevant_papers.append({
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "abstract": paper["abstract"],
                    "core_innovation": paper["core_innovation"],
                    "technical_approach": paper["technical_approach"],
                    "quantitative_findings": paper["quantitative_findings"],
                    "limitations": paper["limitations"],
                    "domain_impact": paper["domain_impact"],
                    "topics": paper["topics"],
                    "score": float(similarity_scores[idx])
                })
            
        return relevant_papers
    
    def initialize_model(self):
        """Initialize LLM for answer generation"""
        if self.model is not None:
            return  # Model already loaded
            
        logger.info(f"Loading model: {self.model_name}")
        
        # Configure for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def answer_question(self, question, top_k=5, temperature=0.7):
        """Answer a research question based on papers"""
        # Find relevant papers
        relevant_papers = self.find_relevant_papers(question, top_k=top_k)
        
        # Add debugging
        logger.info(f"Found {len(relevant_papers)} relevant papers for question: {question}")
        for i, paper in enumerate(relevant_papers):
            logger.info(f"Paper {i+1}: {paper['title']} (Score: {paper['score']:.3f})")
        
        if not relevant_papers:
            logger.warning("No relevant papers found - returning generic response")
            return f"I couldn't find specific research papers about '{question}' in my database. Please try a different question or add relevant papers to the collection."
        
        if self.model is None:
            self.initialize_model()
        
        # Prepare context from relevant papers
        context = ""
        for i, paper in enumerate(relevant_papers):
            context += f"\nPaper {i+1}: \"{paper['title']}\"\n"
            context += f"Authors: {paper['authors']}\n"
            context += f"Abstract: {paper['abstract'][:200]}...\n"
            
            # Add core innovation
            if paper["core_innovation"]:
                context += f"Core Innovation: {paper['core_innovation']}\n"
            
            # Add technical approach
            if paper["technical_approach"]:
                context += f"Technical Approach: {paper['technical_approach']}\n"
            
            # Add findings
            if paper["quantitative_findings"]:
                context += f"Findings: {paper['quantitative_findings']}\n"
            
            # Add limitations
            if paper["limitations"]:
                context += f"Limitations: {paper['limitations']}\n"
            
            # Add domain impact
            if paper["domain_impact"]:
                context += f"Impact: {paper['domain_impact']}\n"
        
        # Prepare prompt with instructions for high-quality answers
        prompt = f"""You are a research assistant specializing in answering questions about scientific papers.
Answer the following question based on the information from these research papers.

Question: "{question}"

Here are the relevant papers:
{context}

Instructions:
1. Provide a comprehensive, factual answer based only on the information in the papers above
2. Include specific technical details and findings from the papers
3. Cite the papers when discussing specific methods or results (e.g., [Paper 1])
4. Explain complex concepts clearly
5. Note any limitations or areas of disagreement among the papers
6. Structure your answer with clear paragraphs

Answer:"""

        # Generate answer
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract only the answer part (after the prompt)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response[len(prompt):].strip()
            
            # Add citations to the answer
            answer += "\n\n**References:**\n"
            for i, paper in enumerate(relevant_papers):
                authors_str = ""
                if isinstance(paper['authors'], list):
                    authors_str = ", ".join(paper['authors']) if paper['authors'] else "Unknown"
                else:
                    authors_str = paper['authors'] if paper['authors'] else "Unknown"
                
                answer += f"[{i+1}] {paper['title']} by {authors_str} (Relevance: {paper['score']:.2f})\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while trying to generate an answer: {str(e)}"
    
    def quick_answer(self, question):
        """Provide a quick answer without using the LLM (for basic questions)"""
        # Find relevant papers
        relevant_papers = self.find_relevant_papers(question, top_k=3)
        
        if not relevant_papers:
            return "No relevant papers found for this query."
        
        # Create a simple response based on the most relevant paper
        top_paper = relevant_papers[0]
        
        response = f"Based on \"{top_paper['title']}\":\n\n"
        
        # Add core innovation
        if top_paper["core_innovation"]:
            response += f"Core innovation: {top_paper['core_innovation']}\n\n"
            
        # Add a brief summary based on technical approach
        if top_paper["technical_approach"]:
            response += f"Technical approach: {top_paper['technical_approach']}\n\n"
            
        # Add key findings
        if top_paper["quantitative_findings"]:
            response += f"Key findings: {top_paper['quantitative_findings']}\n\n"
            
        # Add related papers
        if len(relevant_papers) > 1:
            response += "Other relevant papers include:\n"
            for i, paper in enumerate(relevant_papers[1:], 1):
                response += f"- {paper['title']} (Relevance: {paper['score']:.2f})\n"
        
        return response


# Example usage:
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Question Answerer")
    parser.add_argument("--question", type=str, required=True, help="Research question to answer")
    parser.add_argument("--top_k", type=int, default=5, help="Number of papers to consider")
    parser.add_argument("--quick", action="store_true", help="Use quick answer mode (no LLM)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for answer generation")
    
    args = parser.parse_args()
    
    # Initialize the answerer
    answerer = ResearchQuestionAnswerer()
    
    # Use appropriate answer method
    if args.quick:
        answer = answerer.quick_answer(args.question)
    else:
        answer = answerer.answer_question(
            question=args.question,
            top_k=args.top_k,
            temperature=args.temperature
        )
    
    print(f"\n=== Answer to: {args.question} ===\n")
    print(answer)

if __name__ == "__main__":
    main()