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
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import ListedColormap

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
                    "Words": df["title"],
                    "Frequency": [1] * len(topics)
                })
            )
            self._save_visualization(fig_time, "topics_over_time")
            
            # 5. New text visualizations
            self.visualize_top_topics_text(top_n=10)
            self.visualize_topic_heatmap(top_n=15)
            
            # 6. Cache model
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

    def visualize_top_topics_text(self, top_n: int = 10):
        """Create a text-based visualization showing key words for top topics.
        
        Args:
            top_n: Number of top topics to visualize
        
        Returns:
            Path to the saved visualization
        """
        if not hasattr(self.topic_model, "topics_"):
            logger.warning("Topic model not fitted yet. Run analyze_trends first.")
            return None
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        # Filter for non-outlier topics and get top N
        filtered_topics = topic_info[topic_info['Topic'] != -1].head(top_n)
        
        # Prepare figure
        fig, axes = plt.subplots(int(np.ceil(top_n/2)), 2, figsize=(20, 5*np.ceil(top_n/4)))
        axes = axes.flatten()
        
        # Use a colorful palette for different topics
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Create word clouds for each topic
        for i, (_, row) in enumerate(filtered_topics.iterrows()):
            topic_id = row['Topic']
            
            # Get the words and weights for this topic
            words = dict(self.topic_model.get_topic(topic_id))
            
            if not words:  # Skip if no words for this topic
                continue
                
            # Generate word cloud
            wc = WordCloud(
                background_color='white',
                width=400,
                height=400,
                max_words=15,
                color_func=lambda *args, **kwargs: colors[i % len(colors)],
                prefer_horizontal=1.0
            ).generate_from_frequencies(words)
            
            # Plot on the appropriate subplot
            if i < len(axes):
                axes[i].imshow(wc, interpolation='bilinear')
                axes[i].set_title(f'Topic {topic_id}: {row["Name"]}', fontsize=14)
                axes[i].axis('off')
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        # Save the visualization
        viz_dir = self.cache_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = viz_dir / f"topic_wordcloud_{timestamp}.png"
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved text visualization to {file_path}")
        return file_path

    def visualize_topic_heatmap(self, top_n: int = 15):
        """Create a heatmap showing key terms for top topics.
        
        Args:
            top_n: Number of top topics to visualize
        
        Returns:
            Path to the saved visualization
        """
        if not hasattr(self.topic_model, "topics_"):
            logger.warning("Topic model not fitted yet. Run analyze_trends first.")
            return None
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        # Filter for non-outlier topics and get top N
        filtered_topics = topic_info[topic_info['Topic'] != -1].head(top_n)
        
        # Prepare data for the heatmap
        all_terms = set()
        topic_term_weights = {}
        
        for _, row in filtered_topics.iterrows():
            topic_id = row['Topic']
            topic_terms = self.topic_model.get_topic(topic_id)
            topic_term_weights[topic_id] = dict(topic_terms)
            all_terms.update(dict(topic_terms).keys())
        
        # Create a matrix of term weights per topic
        matrix = []
        terms = sorted(list(all_terms))
        topic_ids = sorted(topic_term_weights.keys())
        
        for topic_id in topic_ids:
            topic_row = []
            for term in terms:
                topic_row.append(topic_term_weights[topic_id].get(term, 0))
            matrix.append(topic_row)
        
        # Create heatmap
        plt.figure(figsize=(20, len(topic_ids) * 0.8))
        ax = sns.heatmap(
            np.array(matrix),
            annot=False,
            xticklabels=terms,
            yticklabels=[f"Topic {tid}" for tid in topic_ids],
            cmap="YlOrRd"
        )
        plt.title("Topic-Term Heatmap", fontsize=16)
        plt.xlabel("Terms", fontsize=14)
        plt.ylabel("Topics", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save the visualization
        viz_dir = self.cache_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = viz_dir / f"topic_heatmap_{timestamp}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved topic heatmap to {file_path}")
        return file_path

    def generate_trends_report(self, top_n_topics: int = 20) -> str:
        """
        Generate a comprehensive text report of trends suitable for LLM analysis.
        
        Args:
            top_n_topics: Number of top topics to include in the report
            
        Returns:
            str: Formatted text report of trends
        """
        if not hasattr(self.topic_model, "topics_"):
            logger.warning("Topic model not fitted yet. Run analyze_trends first.")
            return "Error: Topic model not trained. Please run analyze_trends() first."
        
        # Initialize report sections
        report = []
        report.append("# AI Research Trends Analysis Report\n")
        
        # 1. Model Information
        report.append("## Model Information")
        report.append(f"- Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- Topics identified: {len(self.topic_model.get_topic_info())}")
        report.append("")
        
        # 2. Top Topics Overview
        report.append("## Top Research Topics")
        topic_info = self.topic_model.get_topic_info()
        filtered_topics = topic_info[topic_info['Topic'] != -1].head(top_n_topics)
        
        for i, (_, row) in enumerate(filtered_topics.iterrows()):
            topic_id = row['Topic']
            topic_name = row['Name']
            topic_count = row['Count']
            words = self.topic_model.get_topic(topic_id)
            
            report.append(f"### Topic {topic_id}: {topic_name}")
            report.append(f"- Document count: {topic_count}")
            report.append("- Key terms and weights:")
            for word, weight in words[:10]:  # Top 10 words
                report.append(f"  - {word}: {weight:.4f}")
            report.append("")
        
        # 3. Emerging Topics
        try:
            df = self._fetch_papers()
            df["year"] = df["published"].dt.year
            
            # Get topic distributions per year
            topic_dist = pd.crosstab(df["year"], self.topic_model.topics_)
            
            # Calculate growth rates
            growth_rates = topic_dist.pct_change().rolling(window=2).mean().dropna(how="all")
            
            if not growth_rates.empty:
                report.append("## Emerging Topic Trends")
                
                # Get the most recent year's data
                latest_year = growth_rates.index.max()
                recent_growth = growth_rates.loc[latest_year].sort_values(ascending=False)
                
                # Get top 5 fastest growing topics
                top_growing = recent_growth.nlargest(5).dropna()
                
                for topic_id, growth in top_growing.items():
                    if topic_id == -1:  # Skip outlier topic
                        continue
                        
                    topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].values[0]
                    report.append(f"### Topic {topic_id}: {topic_name}")
                    report.append(f"- Growth rate: {growth:.2%}")
                    report.append("- Key terms:")
                    
                    words = self.topic_model.get_topic(topic_id)
                    for word, _ in words[:5]:
                        report.append(f"  - {word}")
                    report.append("")
                    
                # Get top 5 declining topics
                top_declining = recent_growth.nsmallest(5).dropna()
                
                report.append("### Declining Topics")
                for topic_id, growth in top_declining.items():
                    if topic_id == -1:  # Skip outlier topic
                        continue
                        
                    topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].values[0]
                    report.append(f"- Topic {topic_id} ({topic_name}): {growth:.2%}")
                report.append("")
        except Exception as e:
            logger.error(f"Could not generate emerging topics section: {str(e)}")
            report.append("## Emerging Topic Trends")
            report.append("Could not generate emerging topics analysis due to insufficient time series data.")
            report.append("")
        
        # 4. Topic Coherence and Quality Metrics
        report.append("## Model Quality Metrics")
        
        # Calculate topic diversity (ratio of unique words to total words)
        all_words = []
        unique_words = set()
        
        for topic_id in filtered_topics['Topic']:
            words = self.topic_model.get_topic(topic_id)
            all_words.extend([word for word, _ in words])
            unique_words.update([word for word, _ in words])
        
        topic_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        report.append(f"- Topic diversity: {topic_diversity:.4f}")
        report.append(f"- Number of outlier documents: {topic_info[topic_info['Topic'] == -1]['Count'].values[0]}")
        
        # 5. Suggestions for Analysis
        report.append("## Suggestions for LLM Analysis")
        report.append("1. Are the identified topics coherent and representative of current AI research?")
        report.append("2. Are there any missing important topics that should be present?")
        report.append("3. Do the emerging topics align with known research trends in AI?")
        report.append("4. How could the topic modeling be improved?")
        report.append("5. What search terms would be most effective for tracking these research areas?")
        report.append("")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.cache_dir / f"trend_report_{timestamp}.txt"
        
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Trend report saved to {report_path}")
        
        return "\n".join(report)

def search_empty_or_zero_files(processed_dir: Path):
    """Search for preprocessed files that contain '0', are empty, or have the shortest length"""
    shortest_length = float('inf')
    shortest_files = []

    for json_file in processed_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Check if the text is "0" or empty
                text = data.get("text", "").strip()
                if text == "0" or not text:
                    logger.info(f"Found file with '0' or empty content: {json_file}")
                
                # Compare the length of the text
                text_length = len(text)
                if text_length < shortest_length:
                    shortest_length = text_length
                    shortest_files = [json_file]
                elif text_length == shortest_length:
                    shortest_files.append(json_file)
        
        except Exception as e:
            logger.error(f"Failed to read {json_file}: {e}", exc_info=True)

    # Log the files with the shortest length
    for file in shortest_files:
        logger.info(f"File with shortest length ({shortest_length} characters): {file}")
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
            emerging = analyzer.get_emerging_topics(threshold=0.5)
            print("Emerging Topics:\n", emerging)
            
            # Generate and print the trend report
            trend_report = analyzer.generate_trends_report()
            print("\n--- Trend Report Preview ---")
            print("\n".join(trend_report.split("\n")[:20]) + "\n...")
            print(f"Full report saved to {analyzer.cache_dir}")
        else:
            print("Not enough data for analysis.")

        processed_dir = Path("./Database/processed_pdfs/")
        if not processed_dir.exists():
            logger.error(f"Processed directory {processed_dir} does not exist")
            raise SystemExit(1)
        
        search_empty_or_zero_files(processed_dir)
        print("Search completed")
        
    except TrendAnalysisError as e:
        print(f"Analysis failed: {str(e)}")