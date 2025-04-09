#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModel
from dateutil.parser import parse
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchTimeMachine:
    def __init__(self, papers_dir="./Database/processed_pdfs", highlights_dir="./Database/paper_highlights"):
        self.papers_dir = Path(papers_dir)
        self.highlights_dir = Path(highlights_dir)
        self.papers = {}
        self.paper_timeline = {}
        self.embeddings = {}
        self.tokenizer = None
        self.model = None
        
    def load_papers(self):
        """Load papers with their metadata and highlights for temporal analysis"""
        logger.info("Loading papers for temporal analysis...")
        
        # Load all papers with their metadata
        paper_files = list(self.papers_dir.glob("*.json"))
        highlight_files = list(self.highlights_dir.glob("*_highlights.json"))
        
        # Create a mapping of paper IDs to highlight files
        highlight_map = {}
        for hf in highlight_files:
            paper_id = hf.stem.split("_highlights")[0]
            highlight_map[paper_id] = hf
        
        # Process each paper
        for paper_file in paper_files:
            paper_id = paper_file.stem
            try:
                with open(paper_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Get publication date
                pub_date = None
                if 'metadata' in paper_data and 'published' in paper_data['metadata']:
                    try:
                        pub_date = parse(paper_data['metadata']['published'])
                    except:
                        # Try alternative date formats
                        date_str = paper_data['metadata'].get('published', '')
                        if date_str:
                            try:
                                if date_str.startswith('D:'):
                                    # PDF format date
                                    date_str = date_str[2:14]
                                    pub_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                                else:
                                    # Try common formats
                                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y']:
                                        try:
                                            pub_date = datetime.strptime(date_str[:10], fmt)
                                            break
                                        except:
                                            pass
                            except:
                                pass
                
                # If no publication date, try to infer from file metadata
                if pub_date is None and 'metadata' in paper_data and 'file_mtime' in paper_data['metadata']:
                    pub_date = datetime.fromtimestamp(paper_data['metadata']['file_mtime'])
                
                # Default to current time if all else fails
                if pub_date is None:
                    pub_date = datetime.now()
                
                # Initialize paper entry with temporal data
                self.papers[paper_id] = {
                    'content': paper_data.get('text', ''),
                    'title': paper_data.get('metadata', {}).get('title', paper_id),
                    'authors': paper_data.get('metadata', {}).get('author', 'Unknown'),
                    'pub_date': pub_date,
                    'year': pub_date.year,
                    'highlights': {}
                }
                
                # Store paper in timeline buckets (by year)
                year = pub_date.year
                if year not in self.paper_timeline:
                    self.paper_timeline[year] = []
                self.paper_timeline[year].append(paper_id)
                
                # Add highlights if available
                if paper_id in highlight_map:
                    with open(highlight_map[paper_id], 'r', encoding='utf-8') as f:
                        self.papers[paper_id]['highlights'] = json.load(f)
                
            except Exception as e:
                logger.error(f"Error loading paper {paper_id}: {str(e)}")
        
        # Sort timeline years
        self.years = sorted(self.paper_timeline.keys())
        logger.info(f"Loaded {len(self.papers)} papers spanning from {min(self.years)} to {max(self.years)}")
        
    def init_embedding_model(self):
        """Initialize the embedding model"""
        if self.tokenizer is None or self.model is None:
            logger.info("Initializing embedding model...")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
    def generate_embeddings(self):
        """Generate embeddings for all papers, organized by year"""
        if not self.papers:
            self.load_papers()
            
        self.init_embedding_model()
        logger.info("Generating embeddings for all papers...")
        
        # Create embeddings for all papers
        for paper_id, paper in tqdm(self.papers.items(), desc="Generating embeddings"):
            # Combine title, abstract, and key highlights for embedding
            text_to_embed = f"{paper['title']} "
            
            # Add contribution from highlights if available
            highlights = paper.get('highlights', {})
            if highlights:
                if 'contribution' in highlights:
                    text_to_embed += highlights['contribution'] + " "
                
                # Add methods and results if available
                methods = highlights.get('methods', [])
                if isinstance(methods, list):
                    text_to_embed += " ".join(methods) + " "
                
                results = highlights.get('results', [])
                if isinstance(results, list):
                    text_to_embed += " ".join(results) + " "
                    
                # Add topics if available
                topics = highlights.get('topics', [])
                if isinstance(topics, list):
                    text_to_embed += " ".join(topics) + " "
            
            # Generate embedding
            inputs = self.tokenizer(text_to_embed, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Mean pooling to get sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Store embedding with year information
            year = paper['year']
            if year not in self.embeddings:
                self.embeddings[year] = {}
            
            self.embeddings[year][paper_id] = {
                'embedding': embedding,
                'title': paper['title'],
                'authors': paper['authors'],
                'pub_date': paper['pub_date']
            }
            
        logger.info(f"Generated embeddings for {len(self.papers)} papers across {len(self.embeddings)} years")
        
    def visualize_concept_evolution(self, topic_query, output_path="research_evolution.html"):
        """Visualize how a research concept evolves over time"""
        if not self.embeddings:
            self.generate_embeddings()
        
        logger.info(f"Visualizing evolution of concept: '{topic_query}'")
        
        # Generate embedding for the query
        self.init_embedding_model()
        inputs = self.tokenizer(topic_query, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Find relevant papers across all years using cosine similarity
        relevant_papers = []
        for year, papers in self.embeddings.items():
            for paper_id, paper_data in papers.items():
                similarity = np.dot(query_embedding, paper_data['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(paper_data['embedding']))
                
                if similarity > 0.4:  # Threshold for relevance
                    relevant_papers.append({
                        'paper_id': paper_id,
                        'title': paper_data['title'],
                        'authors': paper_data['authors'],
                        'year': year,
                        'date': paper_data['pub_date'],
                        'similarity': float(similarity),
                        'embedding': paper_data['embedding']
                    })
        
        if not relevant_papers:
            logger.warning(f"No papers found relevant to '{topic_query}'")
            return
            
        logger.info(f"Found {len(relevant_papers)} papers relevant to '{topic_query}'")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(relevant_papers)
        
        # Choose visualization method based on number of papers
        if len(relevant_papers) < 5:
            logger.warning("Too few papers for dimensional reduction visualization")
            # Simple timeline visualization instead
            self._create_simple_timeline(df, topic_query, output_path)
            return
        
        # Perform dimensionality reduction for visualization
        embeddings_array = np.vstack(df['embedding'].values)
        
        # Use PCA first to reduce noise, then t-SNE for visualization
        pca = PCA(n_components=min(50, len(df)-1))
        reduced_embeddings = pca.fit_transform(embeddings_array)
        
        tsne = TSNE(n_components=2, perplexity=min(5, len(df)-1), 
                   learning_rate='auto', init='pca', random_state=42)
        tsne_results = tsne.fit_transform(reduced_embeddings)
        
        df['x'] = tsne_results[:, 0]
        df['y'] = tsne_results[:, 1]
        
        # Create interactive plotly visualization
        fig = px.scatter(
            df, x='x', y='y', color='year', size='similarity',
            hover_name='title', hover_data=['authors', 'year', 'similarity'],
            color_continuous_scale='Viridis', size_max=15,
            title=f'Evolution of Research on "{topic_query}" ({min(df.year)} - {max(df.year)})'
        )
        
        # Add timeline connector lines between consecutive years
        years = sorted(df['year'].unique())
        for i in range(len(years)-1):
            year1, year2 = years[i], years[i+1]
            # Get centroid of each year group
            centroid1 = df[df['year'] == year1][['x', 'y']].mean()
            centroid2 = df[df['year'] == year2][['x', 'y']].mean()
            
            # Draw line between centroids
            fig.add_trace(go.Scatter(
                x=[centroid1['x'], centroid2['x']], 
                y=[centroid1['y'], centroid2['y']],
                mode='lines',
                line=dict(width=2, color='rgba(100,100,100,0.5)'),
                showlegend=False
            ))
            
            # Add year annotations at centroids
            fig.add_annotation(
                x=centroid1['x'], y=centroid1['y'],
                text=str(year1),
                showarrow=False,
                font=dict(size=14)
            )
        
        # Add annotation for last year
        if years:
            last_year = years[-1]
            last_centroid = df[df['year'] == last_year][['x', 'y']].mean()
            fig.add_annotation(
                x=last_centroid['x'], y=last_centroid['y'],
                text=str(last_year),
                showarrow=False,
                font=dict(size=14)
            )
        
        # Save the interactive visualization
        fig.write_html(output_path)
        logger.info(f"Saved visualization to {output_path}")
        
        # Optional: Create a animation of concept evolution through the years
        self._create_concept_evolution_animation(df, topic_query)

    def _create_simple_timeline(self, df, topic_query, output_path):
        """Create a simple timeline visualization when there are few papers"""
        fig = px.scatter(
            df, x='date', y='similarity', color='year',
            hover_name='title', hover_data=['authors', 'similarity'],
            size='similarity', size_max=15,
            title=f'Timeline of Research on "{topic_query}" ({min(df.year)} - {max(df.year)})'
        )
        
        # Add paper titles as annotations
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['date'], 
                y=row['similarity'],
                text=row['title'][:40] + "..." if len(row['title']) > 40 else row['title'],
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            
        fig.write_html(output_path)
        logger.info(f"Saved simple timeline visualization to {output_path}")
        
    def _create_concept_evolution_animation(self, df, topic_query, output_path=None):
        """Create an animation showing how the concept evolved over the years"""
        if output_path is None:
            output_path = f"concept_evolution_{topic_query.replace(' ', '_')}.mp4"
            
        # This is a more advanced visualization that requires more data
        if len(df) < 10 or len(df['year'].unique()) < 3:
            logger.warning("Not enough data for animated visualization")
            return
            
        logger.info("Creating concept evolution animation... (This may take a while)")
        
        # Create animated visualization using matplotlib
        # Simplified animation code would go here
        # This is just a placeholder for a full implementation
        logger.info(f"Animation feature would save to {output_path}")
        logger.info("Animation creation not implemented in this version")

    def predict_future_trends(self, topic_query, years_ahead=2):
        """Predict future research trends related to the query"""
        if not self.embeddings:
            self.generate_embeddings()
            
        logger.info(f"Predicting future trends for '{topic_query}' {years_ahead} years ahead")
        
        # Generate embedding for the query
        self.init_embedding_model()
        inputs = self.tokenizer(topic_query, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Find topic prevalence across years
        topic_trends = {}
        for year, papers in self.embeddings.items():
            year_similarities = []
            for paper_id, paper_data in papers.items():
                similarity = np.dot(query_embedding, paper_data['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(paper_data['embedding']))
                if similarity > 0.4:  # Relevance threshold
                    year_similarities.append(similarity)
            
            # Calculate both prevalence (count) and average relevance
            if year_similarities:
                topic_trends[year] = {
                    'count': len(year_similarities),
                    'avg_relevance': sum(year_similarities) / len(year_similarities),
                    'total_papers': len(papers)
                }
            else:
                topic_trends[year] = {
                    'count': 0,
                    'avg_relevance': 0,
                    'total_papers': len(papers)
                }
        
        # Convert to time series data
        years = sorted(topic_trends.keys())
        
        # We need enough years of data to make a forecast
        if len(years) < 3:
            logger.warning("Not enough historical data for trend forecasting")
            return None
        
        # Time series data: absolute count and relative proportion
        ts_count = [topic_trends[y]['count'] for y in years]
        ts_proportion = [topic_trends[y]['count'] / topic_trends[y]['total_papers'] 
                         if topic_trends[y]['total_papers'] > 0 else 0 for y in years]
        ts_relevance = [topic_trends[y]['avg_relevance'] for y in years]
        
        # Simple forecasting using linear extrapolation
        # For a more sophisticated approach, you could use:
        # - Prophet (Facebook's forecasting tool)
        # - ARIMA/SARIMA models
        # - Transformer-based forecasting models
        
        # Example using linear regression for simplicity
        from sklearn.linear_model import LinearRegression
        
        # Convert years to a format suitable for sklearn
        X = np.array(years).reshape(-1, 1)
        
        # Trend count
        model_count = LinearRegression()
        model_count.fit(X, ts_count)
        future_years = np.array(range(max(years)+1, max(years)+years_ahead+1)).reshape(-1, 1)
        forecast_count = model_count.predict(future_years)
        
        # Trend proportion
        model_prop = LinearRegression()
        model_prop.fit(X, ts_proportion)
        forecast_prop = model_prop.predict(future_years)
        
        # Trend relevance
        model_rel = LinearRegression()
        model_rel.fit(X, ts_relevance)
        forecast_rel = model_rel.predict(future_years)
        
        # Format results
        forecast_results = {
            'historical': {
                'years': years,
                'paper_count': ts_count,
                'proportion': ts_proportion,
                'relevance': ts_relevance
            },
            'forecast': {
                'years': future_years.flatten().tolist(),
                'paper_count': forecast_count.tolist(),
                'proportion': [max(0, min(1, p)) for p in forecast_prop.tolist()],  # Constrain between 0-1
                'relevance': [max(0, min(1, r)) for r in forecast_rel.tolist()]  # Constrain between 0-1
            },
            'trend_direction': 'increasing' if forecast_count[-1] > ts_count[-1] else 'decreasing',
            'confidence': min(0.7, 0.4 + 0.1 * len(years))  # Simple confidence estimate
        }
        
        # Visualize the forecast
        self._visualize_trend_forecast(forecast_results, topic_query)
        
        return forecast_results
        
    def _visualize_trend_forecast(self, forecast, topic_query, output_path=None):
        """Visualize the trend forecast"""
        if output_path is None:
            output_path = f"trend_forecast_{topic_query.replace(' ', '_')}.html"
            
        # Create visualization using plotly
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast['historical']['years'],
            y=forecast['historical']['paper_count'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['forecast']['years'],
            y=forecast['forecast']['paper_count'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence interval (simplified)
        confidence = forecast['confidence']
        lower_bound = [(1-confidence) * y for y in forecast['forecast']['paper_count']]
        upper_bound = [(1+confidence) * y for y in forecast['forecast']['paper_count']]
        
        fig.add_trace(go.Scatter(
            x=forecast['forecast']['years'] + forecast['forecast']['years'][::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'Research Trend Forecast for "{topic_query}"',
            xaxis_title='Year',
            yaxis_title='Number of Relevant Papers',
            legend=dict(x=0.01, y=0.99)
        )
        
        fig.write_html(output_path)
        logger.info(f"Saved trend forecast visualization to {output_path}")

    def generate_alternative_paths(self, topic_query, num_alternatives=3):
        """Generate alternative research paths using the paper database"""
        if not self.embeddings:
            self.generate_embeddings()
        
        logger.info(f"Generating alternative research paths for '{topic_query}'")
        
        # Load your local model for text generation
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # Configure quantization for 4-bit inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load local model (adjust based on your available models)
        try:
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
        
        # Find relevant papers to the topic
        # Generate embedding for the query
        self.init_embedding_model()
        inputs = self.tokenizer(topic_query, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Find top relevant papers
        relevant_papers = []
        for year, papers in self.embeddings.items():
            for paper_id, paper_data in papers.items():
                similarity = np.dot(query_embedding, paper_data['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(paper_data['embedding']))
                
                if similarity > 0.5:  # Higher threshold for better relevance
                    # Get full paper data
                    paper_content = self.papers.get(paper_id, {})
                    highlights = paper_content.get('highlights', {})
                    
                    relevant_papers.append({
                        'paper_id': paper_id,
                        'title': paper_data['title'],
                        'year': year,
                        'similarity': float(similarity),
                        'methods': highlights.get('methods', []),
                        'results': highlights.get('results', []),
                        'limitations': highlights.get('limitations', [])
                    })
        
        # Sort by similarity
        relevant_papers = sorted(relevant_papers, key=lambda x: x['similarity'], reverse=True)[:10]
        
        if not relevant_papers:
            logger.warning(f"No papers found relevant to '{topic_query}'")
            return None
        
        # Extract key elements for alternative path generation
        methods = []
        limitations = []
        for paper in relevant_papers:
            if isinstance(paper['methods'], list):
                methods.extend(paper['methods'])
            if isinstance(paper['limitations'], list):
                limitations.extend(paper['limitations'])
        
        # Create a context from the relevant papers
        context = f"Topic: {topic_query}\n\n"
        context += "Current Research Approaches:\n"
        for i, method in enumerate(methods[:5]):
            context += f"- {method}\n"
        
        context += "\nCurrent Research Limitations:\n"
        for i, limitation in enumerate(limitations[:5]):
            context += f"- {limitation}\n"
        
        # Generate alternative research paths
        alternatives = []
        for i in range(num_alternatives):
            # Prompt engineering for alternative path generation
            prompt = f"""{context}

Based on the research topic "{topic_query}" and the current approaches and limitations described above, generate a detailed alternative research direction that hasn't been fully explored yet.

The alternative research direction should include:
1. A novel approach that addresses current limitations
2. Key methods that would be used
3. Expected results and impact
4. Required technologies or innovations

Alternative Research Direction {i+1}:"""

            # Generate the alternative path
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,  # Higher temperature for more creativity
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            alternative_text = generated_text[len(prompt):]
            
            # Structure the alternative path
            # Parse the generated text into structured format
            import re
            
            # Extract key components with regex
            approach_match = re.search(r"Novel Approach:(.+?)(?=\n\d\.|\nKey Methods|\n\n)", alternative_text, re.DOTALL)
            methods_match = re.search(r"Key Methods:(.+?)(?=\n\d\.|\nExpected Results|\n\n)", alternative_text, re.DOTALL)
            results_match = re.search(r"Expected Results:(.+?)(?=\n\d\.|\nRequired Technologies|\n\n)", alternative_text, re.DOTALL)
            tech_match = re.search(r"Required Technologies:(.+?)(?=$|\n\n)", alternative_text, re.DOTALL)
            
            structured_alternative = {
                "path_id": i+1,
                "approach": approach_match.group(1).strip() if approach_match else "Not specified",
                "methods": [m.strip() for m in methods_match.group(1).strip().split('\n-')] if methods_match else [],
                "expected_results": results_match.group(1).strip() if results_match else "Not specified",
                "required_technologies": tech_match.group(1).strip() if tech_match else "Not specified",
                "full_text": alternative_text.strip()
            }
            
            alternatives.append(structured_alternative)
        
        # Save the alternatives
        output_file = f"alternative_paths_{topic_query.replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alternatives, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Generated {len(alternatives)} alternative research paths and saved to {output_file}")
        
        return alternatives

# Example usage in main function
def main():
    parser = argparse.ArgumentParser(description="Research Time Machine")
    parser.add_argument("--query", type=str, required=True, help="Research topic to analyze")
    parser.add_argument("--mode", type=str, default="evolution", 
                        choices=["evolution", "forecast", "alternative"],
                        help="Analysis mode: evolution, forecast, or alternative paths")
    parser.add_argument("--years-ahead", type=int, default=3, help="Years to forecast ahead")
    parser.add_argument("--alternatives", type=int, default=3, help="Number of alternative paths to generate")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    time_machine = ResearchTimeMachine()
    time_machine.load_papers()
    
    if args.mode == "evolution":
        output = args.output or f"research_evolution_{args.query.replace(' ', '_')}.html"
        time_machine.visualize_concept_evolution(args.query, output)
        print(f"Visualization saved to {output}")
        
    elif args.mode == "forecast":
        forecast = time_machine.predict_future_trends(args.query, args.years_ahead)
        if forecast:
            direction = forecast['trend_direction']
            confidence = forecast['confidence'] * 100
            print(f"Research on '{args.query}' is predicted to be {direction} with {confidence:.1f}% confidence")
            print(f"Forecast visualization saved")
            
    elif args.mode == "alternative":
        alternatives = time_machine.generate_alternative_paths(args.query, args.alternatives)
        if alternatives:
            print(f"Generated {len(alternatives)} alternative research paths")
            for i, alt in enumerate(alternatives):
                print(f"\nAlternative Path {i+1}:")
                print(f"Approach: {alt['approach'][:100]}...")
    
if __name__ == "__main__":
    main()