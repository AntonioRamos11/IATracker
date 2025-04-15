import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from pathlib import Path
import os
import logging
from typing import Dict, List, Any
import matplotlib.cm as cm



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from .trending_topics import (
    scrape_trending_topics, 
    analyze_topic_coverage,
    get_emerging_topics,
    load_valid_topics
)

class TrendVisualizer:
    """Visualization tools for arXiv trending topics"""
    
    def __init__(self, output_dir=None):
        """Initialize visualizer"""
        if output_dir is None:
            self.output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "trend_visualizations"
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.figsize': (12, 8)
        })
    
    def visualize_top_trends(self, trend_results, top_n=15, save=True, show=True):
        """
        Create bar chart of top trending topics by score.
        
        Args:
            trend_results: Output from identify_emerging_trends()
            top_n: Number of top trends to display
            save: Whether to save the plot to file
            show: Whether to display the plot
            
        Returns:
            Path to saved image if save=True, otherwise None
        """
        # Extract data
        top_trends = trend_results.get('top_trends', [])[:top_n]
        
        if not top_trends:
            logger.warning("No trend data to visualize")
            return None
        
        # Prepare data for plotting
        terms = [t['term'] for t in top_trends]
        scores = [t['score'] for t in top_trends]
        growth_rates = [t['growth'] for t in top_trends]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # Score plot
        score_df = pd.DataFrame({'Term': terms, 'Score': scores})
        score_df = score_df.sort_values('Score', ascending=True)
        
        sns.barplot(x='Score', y='Term', data=score_df, palette='viridis', ax=ax1)
        ax1.set_title('Top Trending Topics by Overall Score')
        ax1.set_xlabel('Score (Frequency × Growth)')
        ax1.set_ylabel('')
        
        # Growth rate plot
        growth_df = pd.DataFrame({'Term': terms, 'Growth Rate': growth_rates})
        growth_df = growth_df.sort_values('Growth Rate', ascending=True)
        
        sns.barplot(x='Growth Rate', y='Term', data=growth_df, palette='rocket', ax=ax2)
        ax2.set_title('Top Trending Topics by Growth Rate')
        ax2.set_xlabel('Growth Rate')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        # Save figure
        output_path = None
        if save:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"top_trends_{timestamp}.png"
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved trend visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path
    
    def create_topic_wordcloud(self, trend_results, save=True, show=True):
        """
        Generate a word cloud from trending topics.
        
        Args:
            trend_results: Output from identify_emerging_trends()
            save: Whether to save the word cloud to file
            show: Whether to display the word cloud
            
        Returns:
            Path to saved image if save=True, otherwise None
        """
        # Extract data
        trends = trend_results.get('top_trends', [])
        
        if not trends:
            logger.warning("No trend data for word cloud")
            return None
        
        # Create word frequency dictionary
        word_freq = {}
        for trend in trends:
            word_freq[trend['term']] = trend['score']
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1000, 
            height=600, 
            background_color='white',
            colormap='viridis',
            min_font_size=10,
            max_font_size=150,
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        # Plot
        plt.figure(figsize=(16, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Trending Research Topics Word Cloud', fontsize=20, pad=20)
        
        # Save figure
        output_path = None
        if save:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"wordcloud_{timestamp}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved word cloud to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path
    
    def visualize_topic_coverage(self, coverage_data, save=True, show=True):
        """
        Visualize topic coverage from analyze_topic_coverage().
        
        Args:
            coverage_data: Output from analyze_topic_coverage()
            save: Whether to save the visualization
            show: Whether to show the visualization
            
        Returns:
            Path to saved image if save=True, otherwise None
        """
        if not coverage_data:
            logger.warning("No coverage data to visualize")
            return None
            
        # Extract data for plotting
        topics = []
        paper_counts = []
        descriptions = []
        
        for topic, data in coverage_data.items():
            topics.append(topic)
            paper_counts.append(data['total'])
            descriptions.append(data.get('description', topic))
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Topic': topics,
            'Description': descriptions,
            'Papers': paper_counts
        })
        
        # Sort by paper count
        df = df.sort_values('Papers', ascending=False)
        
        # Create plots - both pie and bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Pie chart
        plt.sca(ax1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        wedges, texts, autotexts = ax1.pie(
            df['Papers'], 
            autopct='%1.1f%%',
            shadow=False, 
            startangle=90,
            colors=colors
        )
        ax1.set_title('Distribution of Papers by Topic')
        
        # Add legend with topic descriptions
        shortened_descriptions = [d[:30] + '...' if len(d) > 30 else d for d in df['Description']]
        ax1.legend(
            wedges, 
            shortened_descriptions,
            title="Topics",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Bar chart
        plt.sca(ax2)
        sns.barplot(x='Papers', y='Description', data=df.head(15), palette='viridis', ax=ax2)
        ax2.set_title('Top Topics by Number of Papers')
        ax2.set_xlabel('Number of Papers')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        # Save figure
        output_path = None
        if save:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"topic_coverage_{timestamp}.png"
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved coverage visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path
    
    def visualize_topic_papers_by_year(self, coverage_data, save=True, show=True):
        """
        Visualize distribution of papers by year for each topic.
        
        Args:
            coverage_data: Output from analyze_topic_coverage()
            save: Whether to save the visualization
            show: Whether to show the visualization
            
        Returns:
            Path to saved image if save=True, otherwise None
        """
        if not coverage_data:
            logger.warning("No coverage data to visualize")
            return None
        
        # Prepare data for each topic
        topics_with_year_data = []
        for topic, data in coverage_data.items():
            if data.get('by_year') and len(data['by_year']) > 0:
                topics_with_year_data.append((
                    data.get('description', topic),
                    data.get('by_year', {})
                ))
        
        if not topics_with_year_data:
            logger.warning("No year distribution data available")
            return None
        
        # Create a plot per topic
        n_topics = len(topics_with_year_data)
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        fig.suptitle('Papers by Year for Each Topic', fontsize=20)
        
        # Ensure axes is always a 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (topic, year_data) in enumerate(topics_with_year_data):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            years = list(year_data.keys())
            counts = list(year_data.values())
            
            # Create DataFrame for plotting
            year_df = pd.DataFrame({'Year': years, 'Papers': counts})
            year_df = year_df.sort_values('Year')
            
            # Plot
            sns.barplot(x='Year', y='Papers', data=year_df, ax=ax, palette='viridis')
            ax.set_title(f"{topic}")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Papers')
            
            # Rotate x-axis labels for readability
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Hide unused subplots
        for i in range(len(topics_with_year_data), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = None
        if save:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"papers_by_year_{timestamp}.png"
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            logger.info(f"Saved year distribution visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return output_path

# Example usage function
def visualize_trending_topics(trend_results=None, coverage_data=None):
    """
    Create visualizations for trending topics.
    
    Args:
        trend_results: Results from identify_emerging_trends() (optional)
        coverage_data: Results from analyze_topic_coverage() (optional)
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    from trending_topics import identify_emerging_trends, analyze_topic_coverage
    
    visualizer = TrendVisualizer()
    output_paths = {}
    
    # Get trend data if not provided
    if trend_results is None:
        try:
            logger.info("Fetching trending topics data...")
            trend_results = identify_emerging_trends(
                categories=["cs.AI", "cs.LG", "cs.CL"],
                initial_sample_size=200
            )
        except Exception as e:
            logger.error(f"Error fetching trend data: {e}")
            trend_results = None
    
    # Create trend visualizations
    if trend_results:
        # Top trends bar chart
        bar_path = visualizer.visualize_top_trends(
            trend_results, 
            top_n=15,
            show=False
        )
        output_paths['top_trends_chart'] = bar_path
        
        # Word cloud
        wc_path = visualizer.create_topic_wordcloud(
            trend_results,
            show=False
        )
        output_paths['wordcloud'] = wc_path
    
    # Create coverage visualizations
    if coverage_data:
        # Topic distribution
        dist_path = visualizer.visualize_topic_coverage(
            coverage_data,
            show=False
        )
        output_paths['topic_distribution'] = dist_path
        
        # Papers by year
        year_path = visualizer.visualize_topic_papers_by_year(
            coverage_data,
            show=False
        )
        output_paths['papers_by_year'] = year_path
    
    return output_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trending topics in arXiv papers")
    parser.add_argument("--load-data", action="store_true", help="Load trend data instead of fetching new data")
    parser.add_argument("--save-dir", type=str, help="Directory to save visualizations")
    args = parser.parse_args()
    
    save_dir = args.save_dir if args.save_dir else None
    
    try:
        # Import necessary functions
        from trending_topics import identify_emerging_trends, analyze_topic_coverage, scrape_trending_topics
        
        # Get the data
        if args.load_data:
            # Load from saved files if available
            import json
            from pathlib import Path
            
            data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "trend_cache"
            trend_file = data_dir / "latest_trends.json"
            coverage_file = data_dir / "topic_coverage.json"
            
            trend_results = None
            coverage_data = None
            
            if trend_file.exists():
                with open(trend_file, 'r') as f:
                    trend_results = json.load(f)
                    print(f"Loaded trend data from {trend_file}")
            
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    print(f"Loaded coverage data from {coverage_file}")
        else:
            # Get fresh data
            print("Identifying emerging trends...")
            trend_results = identify_emerging_trends(
                categories=["cs.AI", "cs.LG", "cs.CL"],
                initial_sample_size=200
            )
            
            print("Getting papers for trending topics...")
            topics = get_emerging_topics(categories=["cs.AI", "cs.LG", "cs.CL"])
            papers = scrape_trending_topics(
                max_results_per_topic=50,
                batch_size=10,
                custom_topics=topics
            )
            
            print("Analyzing topic coverage...")
            coverage_data = analyze_topic_coverage(papers)
        
        # Create visualizer and generate visualizations
        visualizer = TrendVisualizer(output_dir=save_dir)
        
        if trend_results:
            print("\nGenerating trend visualizations...")
            visualizer.visualize_top_trends(trend_results)
            visualizer.create_topic_wordcloud(trend_results)
        
        if coverage_data:
            print("\nGenerating coverage visualizations...")
            visualizer.visualize_topic_coverage(coverage_data)
            visualizer.visualize_topic_papers_by_year(coverage_data)
        
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()