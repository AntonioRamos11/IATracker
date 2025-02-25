
import ast
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from bertopic import BERTopic
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pathlib import Path
import joblib
import psycopg2
from psycopg2 import sql, errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrendAnalysisError(Exception):
    """Base exception for trend analysis errors"""

class TrendAnalyzer:
    def __init__(self, db_config: Dict, cache_dir: str = "./trend_cache"):
        self.db_config = db_config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.topic_model = None
        self._prepare_environment()

    def _prepare_environment(self):
        """Validate environment dependencies"""
        try:
            import bertopic
            import umap
        except ImportError as e:
            logger.error("Missing required libraries: %s", str(e))
            raise TrendAnalysisError("Dependencies missing") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((errors.OperationalError, errors.InterfaceError))
    )
    def _fetch_papers(self, batch_size: int = 1000) -> pd.DataFrame:
        """Fetch papers from database with batch processing"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            papers = []
            cursor.execute("SELECT title, metadata->>'published', embedding FROM paper_vectors")
            while batch := cursor.fetchmany(batch_size):
                papers.extend(batch)
            
            logger.info(f"Fetched {len(papers)} papers from the database")
            
            # Log the content of the papers list
            logger.debug(f"Papers content: {papers}")
            
            df = pd.DataFrame(papers, columns=["title", "published", "embedding"])
            df["published"] = pd.to_datetime(df["published"], errors="coerce")
            
            # Log the content of the DataFrame before dropping NaNs
            logger.debug(f"DataFrame content before dropping NaNs:\n{df}")
            
            logger.info(f"DataFrame shape before dropping NaNs: {df.shape}")
            
            # Check for NaN values in the embedding column
            logger.debug(f"NaN values in embedding column: {df['embedding'].isna().sum()}")
            
            df = df.dropna(subset=["published", "embedding"])
            logger.info(f"DataFrame shape after dropping NaNs: {df.shape}")
            
            # Log the content of the DataFrame after dropping NaNs
            logger.debug(f"DataFrame content after dropping NaNs:\n{df}")
            
            return df
            
        except Exception as e:
            logger.error("Database fetch failed: %s", str(e))
            raise TrendAnalysisError("Data retrieval failed") from e
        finally:
            cursor.close()
            conn.close()
    def _validate_embeddings(self, embeddings: List) -> np.ndarray:
        """Validate and normalize embeddings"""
        parsed_embeddings = []
        for emb in embeddings:
            # If the embedding is a string, parse it into a list
            if isinstance(emb, str):
                try:
                    parsed_emb = ast.literal_eval(emb)
                except Exception as e:
                    raise TrendAnalysisError(f"Failed to parse embedding: {emb}") from e
            else:
                parsed_emb = emb
            parsed_embeddings.append(parsed_emb)
        
        embeddings_arr = np.array(parsed_embeddings, dtype=np.float32)
        
        if embeddings_arr.ndim != 2:
            raise TrendAnalysisError(f"Invalid embedding shape: {embeddings_arr.shape}")
            
        # Normalize embeddings
        scaler = MinMaxScaler()
        return scaler.fit_transform(embeddings_arr) 

    def _get_topic_model(self, force_refresh: bool = False) -> BERTopic:
        """Load or create topic model with caching"""
        model_path = self.cache_dir / "topic_model.joblib"
        
        if not force_refresh and model_path.exists():
            logger.info("Loading cached topic model")
            self.topic_model = joblib.load(model_path)
            return self.topic_model
            
        logger.info("Training new topic model")
        umap_model = UMAP(n_components=5, random_state=42)
        self.topic_model = BERTopic(
            umap_model=umap_model,
            language="english",
            calculate_probabilities=True,
            verbose=True
        )
        return self.topic_model

    def _save_visualization(self, fig, name: str):
        """Save visualization with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.cache_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        file_path = viz_dir / f"{name}_{timestamp}.html"
        fig.write_html(file_path)
        logger.info("Saved visualization: %s", file_path)

    def analyze_trends(self, years: Optional[List[int]] = None, force_refresh: bool = False):
        """Main analysis pipeline"""
        try:
            # 1. Data Acquisition
            df = self._fetch_papers()
            
            if years:
                df = df[df["published"].dt.year.isin(years)]
                
            if len(df) < 100:
                logger.warning("Insufficient data for analysis (%d documents)", len(df))
                return

            # 2. Embedding Validation
            embeddings = self._validate_embeddings(df["embedding"].tolist())
            
            # 3. Model Training
            topic_model = self._get_topic_model(force_refresh)
            topics, _ = topic_model.fit_transform(df["title"], embeddings)
            
            # 4. Visualization
            fig = topic_model.visualize_barchart(top_n_topics=20, n_words=10)
            self._save_visualization(fig, "topic_barchart")
            
            fig_time = topic_model.visualize_topics_over_time(
                topics_over_time=pd.DataFrame({
                    "Topic": topics,
                    "Timestamp": df["published"],
                    "Words": df["title"],#added 
                    "Frequency": [1] * len(topics)  # Add a Frequency column with a default value of 1

                })
            )
            self._save_visualization(fig_time, "topics_over_time")
            
            # 5. Cache model
            joblib.dump(topic_model, self.cache_dir / "topic_model.joblib")
            
            return topic_model.get_topic_info()
            
        except Exception as e:
            logger.error("Trend analysis failed: %s", str(e), exc_info=True)
            raise TrendAnalysisError("Analysis pipeline failed") from e

    def get_emerging_topics(self, threshold: float = 0.5) -> pd.DataFrame:
        """Identify emerging topics using rate of change"""
        try:
            df = self._fetch_papers()
            df["year"] = df["published"].dt.year
            embeddings = self._validate_embeddings(df["embedding"].tolist())
            topic_model = self._get_topic_model()
            
            # Check if the model is fitted; BERTopic sets the attribute 'topics_' once fitted
            if not hasattr(topic_model, "topics_"):
                logger.info("BERTopic model is not fitted yet. Fitting now.")
                topics, _ = topic_model.fit_transform(df["title"], embeddings)
            else:
                topics, _ = topic_model.transform(df["title"], embeddings)
            
            # Get topic distributions per year
            topic_dist = pd.crosstab(df["year"], topics)
            logger.debug(f"Topic distributions per year:\n{topic_dist}")
            
            growth_rates = topic_dist.pct_change().rolling(window=3).mean()
            logger.debug(f"Growth rates:\n{growth_rates}")
            
            emerging_topics = growth_rates[growth_rates > threshold].dropna(how="all")
            logger.debug(f"Emerging topics:\n{emerging_topics}")
            
            return emerging_topics
            
        except Exception as e:
            logger.error("Emerging topic detection failed: %s", str(e))
            raise TrendAnalysisError("Emerging topic analysis failed") from e

if __name__ == "__main__":
    db_config = {
        "dbname": "ai_papers",
        "user": "postgres",
        "password": "Chappie1101",
        "host": "localhost",
        "port": 5432
    }
    
    try:
        analyzer = TrendAnalyzer(db_config)
        
        # Analyze papers from last 5 years
        current_year = datetime.now().year
        results = analyzer.analyze_trends(years=list(range(current_year-5, current_year+1)))
        if results is not None:
            emerging = analyzer.get_emerging_topics()
            print("Emerging Topics:\n", emerging)
        else:
            print("Not enough data for analysis.")

        
    except TrendAnalysisError as e:
        print(f"Analysis failed: {str(e)}")