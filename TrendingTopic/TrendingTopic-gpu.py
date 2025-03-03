import torch
import json
from pathlib import Path
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP  # GPU-accelerated UMAP
from datetime import datetime
import logging

# Configure GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchTopicAnalyzer:
    def __init__(self, processed_dir: str = "./Database/processed_pdfs/"):
        self.processed_dir = Path(processed_dir)
        self.embedder = SentenceTransformer("all-mpnet-base-v2").to(device)
        self.topic_model = None
        self.df = None
        
    def load_processed_data(self) -> pd.DataFrame:
        """Load all processed JSON files into a DataFrame"""
        papers = []
        for json_file in self.processed_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    papers.append({
                        "text": data["text"],
                        "title": data["metadata"]["title"],
                        "published": data["metadata"]["published"],
                        "arxiv_id": data["metadata"]["arxiv_id"]
                    })
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
                
        self.df = pd.DataFrame(papers)
        self.df["published"] = pd.to_datetime(self.df["published"], errors="coerce")
        return self.df.dropna(subset=["published"])
        
    def train_topic_model(self):
        """Train BERTopic model on processed documents with GPU acceleration"""
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Call load_processed_data() first")
            
        # Generate embeddings on GPU
        embeddings = self.embedder.encode(
            self.df["text"].tolist(),
            show_progress_bar=True,
            device=device,
            convert_to_numpy=False  # Keep as torch tensor
        ).cpu().numpy()  # Move to CPU numpy array for compatibility
        
        # GPU-accelerated UMAP
        umap_model = UMAP(n_components=5, random_state=42)
        
        # BERTopic with GPU-optimized components
        self.topic_model = BERTopic(
            n_gram_range=(1, 2),
            min_topic_size=15,
            umap_model=umap_model,
            verbose=True
        )
        
        # Convert to lists for GPU compatibility
        docs = self.df["text"].tolist()
        timestamps = self.df["published"].tolist()
        
        # Train model with GPU-accelerated components
        topics, _ = self.topic_model.fit_transform(
            documents=docs,
            embeddings=embeddings
        )
        
        # Temporal analysis with GPU support
        self.topic_model.topics_over_time(
            docs=docs,  
            timestamps=timestamps,
            global_tuning=True,
            evolution_tuning=True
        )
        
        return topics

    # ... (keep other methods unchanged)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Using CPU instead.")
        
    analyzer = ResearchTopicAnalyzer()
    
    # Load preprocessed data
    analyzer.load_processed_data()
    
    # Train model with GPU acceleration
    with torch.cuda.amp.autocast():  # Mixed precision training
       top_topics= analyzer.train_topic_model()
    
    # ... (rest of main code remains the same)  top_topics = analyzer.get_top_topics(10)
    print("Top Trending Topics:")
    print(top_topics[["Topic", "Name", "Frequency", "growth_rate"]])
    
    # Generate visualization
    fig = analyzer.visualize_trends()
    fig.write_html("research_trends.html")