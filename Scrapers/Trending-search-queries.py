import json
from pathlib import Path
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchTopicAnalyzer:
    def __init__(self, processed_dir: str = "./Database/processed_pdfs/"):
        self.processed_dir = Path(processed_dir)
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.topic_model = None
        self.df = None

    def load_processed_data(self) -> pd.DataFrame:
        """Load and validate processed JSON files"""
        papers = []
        valid_count = 0
        
        for json_file in self.processed_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    
                    # Validate required fields
                    if not all(key in data for key in ["text", "metadata"]):
                        logger.warning(f"Skipping invalid file: {json_file}")
                        continue
                        
                    text = data["text"].strip()
                    published = data["metadata"].get("published")
                    
                    # Filter empty texts and invalid dates
                    if len(text) < 100 or not published:
                        continue
                        
                    papers.append({
                        "text": text,
                        "title": data["metadata"].get("title", "Untitled"),
                        "published": published,
                        "arxiv_id": data["metadata"].get("arxiv_id")
                    })
                    valid_count += 1
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
        
        logger.info(f"Loaded {valid_count} valid papers from {len(papers)} files")
        
        if not papers:
            raise ValueError("No valid papers found in processed data")
            
        self.df = pd.DataFrame(papers)
        
        # Convert dates with enhanced validation
        self.df["published"] = pd.to_datetime(
            self.df["published"], 
            errors="coerce",
            format="mixed"
        )
        
        # Filter invalid dates
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=["published"])
        filtered_count = initial_count - len(self.df)
        
        logger.info(f"Filtered {filtered_count} papers with invalid dates")
        
        if self.df.empty:
            raise ValueError("No papers with valid dates remaining")
            
        return self.df

    def train_topic_model(self):
        """Train model with enhanced validation"""
        if self.df is None or self.df.empty:
            raise ValueError("No valid data available for training")
            
        # Check text length distribution
        text_lengths = self.df["text"].str.len()
        logger.info(f"Text length stats:\n{text_lengths.describe()}")
        
        # Filter very short texts
        self.df = self.df[text_lengths >= 500]
        
        if self.df.empty:
            raise ValueError("No papers with sufficient text content")
            
        # Generate embeddings
        try:
            embeddings = self.embedder.encode(
                self.df["text"].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
            
        # Verify embeddings
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings generated - check input texts")
            
        # Initialize BERTopic
        self.topic_model = BERTopic(
            n_gram_range=(1, 2),
            min_topic_size=15,
            verbose=True
        )
        
        # Train model
        try:
            topics, _ = self.topic_model.fit_transform(
                documents=self.df["text"].tolist(),
                embeddings=embeddings
            )
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
            
        # Temporal analysis
        self.df["published"] = pd.to_datetime(self.df["published"], errors="coerce")
        self.df = self.df.dropna(subset=["published"])  # Drop invalid dates
        self.df["timestamp"] = self.df["published"].astype(int) // 10**9
        try:
            topics_over_time = self.topic_model.topics_over_time(
                docs=self.df["text"],
                timestamps=self.df["timestamp"],  # Use correct timestamps
                global_tuning=True,
                evolution_tuning=True,
                nr_bins=20
            )
            print("Available topics in topics_over_time:", topics_over_time["Topic"].unique())

        except KeyError as e:
            print(f"KeyError encountered: {e}")
            print("Possible causes: no valid topics, incorrect timestamps, or all topics are outliers.")
            print("Check topic counts:", pd.Series(self.topic_model.topics_).value_counts())
            raise
        except Exception as e:
            logger.error(f"Temporal analysis failed: {str(e)}")
            raise
            
        return topics

    # Rest of the class remains the same

    def get_top_topics(self, n_topics: int = 10) -> pd.DataFrame:
        """Get top trending topics with temporal information"""
        if self.topic_model is None:
            raise ValueError("Model not trained. Call train_topic_model() first")
        if  self.topic_model.topics_ is None:
            raise ValueError("No topics found. Call train_topic_model() first") 
      
        topics_over_time = self.topic_model.topics_over_time(
            docs=self.df["text"],
            timestamps=self.df["timestamp"],
            global_tuning=True,
            evolution_tuning=True,
            nr_bins=20
        )
        
        # Calculate growth metrics
        topics_over_time["growth_rate"] = topics_over_time.groupby("Topic")["Frequency"].pct_change()
        topics_over_time["momentum"] = topics_over_time["growth_rate"].rolling(3).mean()
        
        # Get most recent topics
        recent_topics = topics_over_time[
            topics_over_time["Timestamp"] == topics_over_time["Timestamp"].max()
        ].nlargest(n_topics, "Frequency")
        
        return recent_topics

    def visualize_trends(self):
        """Generate interactive visualization of topics over time"""
        if self.topic_model is None:
            raise ValueError("Model not trained. Call train_topic_model() first")
            
        return self.topic_model.visualize_topics_over_time(
            self.topic_model.topics_over_time_,
            top_n_topics=20
        )

# Usage example
if __name__ == "__main__":
    analyzer = ResearchTopicAnalyzer()
    
    # Load preprocessed data
    analyzer.load_processed_data()
    
    # Train model
    analyzer.train_topic_model()
    
    # Get top 10 trending topics
    top_topics = analyzer.get_top_topics(10)
    print("Top Trending Topics:")
    print(top_topics[["Topic", "Name", "Frequency", "growth_rate"]])
    
    # Generate visualization
    fig = analyzer.visualize_trends()
    fig.write_html("research_trends.html")