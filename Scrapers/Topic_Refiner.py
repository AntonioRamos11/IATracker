from keybert import KeyBERT
import re
from pathlib import Path
import json
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

class TopicRefiner:
    def __init__(self):
        self.kw_model = KeyBERT()
        self.topic_titles = {}
        
    def clean_topic_name(self, topic_name):
        """Clean topic names to remove IDs and separate CamelCase"""
        # Remove numeric prefix like "0_"
        topic_name = re.sub(r'^\d+_', '', topic_name)
        
        # Remove hash-like strings
        topic_name = re.sub(r'[a-f0-9]{10,}', '', topic_name)
        
        # Split CamelCase
        topic_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', topic_name)
        
        # Remove underscores and extra spaces
        topic_name = topic_name.replace('_', ' ').strip()
        
        return topic_name
    
    def generate_topic_title(self, topic_id, topic_name, sample_texts):
        """Generate human-readable topic titles from document samples"""
        
        # Combine all sample texts
        combined_text = " ".join(sample_texts)
        
        # Extract keywords with KeyBERT
        keywords = self.kw_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            use_mmr=True,
            diversity=0.7
        )
        
        # Get the top 3-4 keywords
        top_keywords = [kw for kw, _ in keywords[:4]]
        
        # Create a descriptive title
        if top_keywords:
            # Capitalize each word
            capitalized_keywords = [word.title() for word in top_keywords]
            title = " & ".join(capitalized_keywords)
        else:
            # Fallback to cleaned topic name if KeyBERT didn't find good keywords
            cleaned_name = self.clean_topic_name(topic_name)
            words = cleaned_name.split()
            title = " ".join([word.title() for word in words[:4]])
        
        self.topic_titles[topic_id] = title
        return title
    
    def refine_topics_from_report(self, report_path, corpus_path=None):
        """Process a trend report and generate better topic titles"""
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Extract topic sections from report
        topic_pattern = r'### Topic (\d+): ([^\n]+)\n- Document count: (\d+)\n- Key terms and weights:(.*?)(?=\n\n|\Z)'
        topics = re.findall(topic_pattern, content, re.DOTALL)
        
        results = []
        
        # If we have access to the original corpus, use it
        corpus_docs = {}
        if corpus_path:
            # Load corpus documents
            if corpus_path.endswith('.json'):
                with open(corpus_path, 'r') as f:
                    corpus_data = json.load(f)
                    if isinstance(corpus_data, list):
                        corpus_docs = {i: doc.get('text', '') for i, doc in enumerate(corpus_data)}
            elif corpus_path.endswith('.csv'):
                df = pd.read_csv(corpus_path)
                if 'text' in df.columns:
                    corpus_docs = {i: text for i, text in enumerate(df['text'])}
        
        for topic_id, topic_name, doc_count, terms_section in topics:
            # Extract terms and weights
            term_lines = re.findall(r'- ([^:]+): ([0-9.]+)', terms_section)
            valid_terms = [(term.strip(), float(weight)) 
                          for term, weight in term_lines 
                          if len(term) < 30 and not term.endswith('.pdf')]
            
            # Get sample texts if available
            sample_texts = []
            if corpus_docs:
                # Find documents that belong to this topic
                # This would require the topic assignments from BERTopic
                # As a fallback, use the extracted terms
                for term, _ in valid_terms[:5]:
                    sample_texts.append(f"{term} is an important concept in this topic.")
            else:
                # Use terms as sample texts
                sample_texts = [term for term, _ in valid_terms[:10]]
            
            # Generate human-readable title
            refined_title = self.generate_topic_title(topic_id, topic_name, sample_texts)
            
            results.append({
                'original_id': topic_id,
                'original_name': topic_name,
                'refined_title': refined_title,
                'document_count': doc_count,
                'top_terms': [term for term, _ in valid_terms[:5]]
            })
            
        return results

# Example usage
if __name__ == "__main__":
    refiner = TopicRefiner()
    
    # Process trend report
    report_path = Path("../trend_cache/trend_report_20250304_232626.txt")
    
    if report_path.exists():
        refined_topics = refiner.refine_topics_from_report(report_path)
        
        print("Original vs Refined Topic Names:")
        for topic in refined_topics:
            print(f"Topic {topic['original_id']}: {topic['original_name']} -> {topic['refined_title']}")
            print(f"  Top terms: {', '.join(topic['top_terms'])}")
            print()
    else:
        print(f"Report file not found: {report_path}")