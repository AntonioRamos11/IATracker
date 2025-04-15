from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import sys
import logging
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from Scrapers.arxiv import get_arxiv_papers
    from Scrapers.topic_processor import TopicProcessor
    from Scrapers.trending_topics import (
        identify_emerging_trends,
        get_emerging_topics,
        scrape_trending_topics,
        analyze_topic_coverage
    )
    from Scrapers.visualize_trends import TrendVisualizer
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webapp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global variables
topic_processor = TopicProcessor()
visualizer = TrendVisualizer(output_dir='webapp/static/images/generated')

# Cache for trend data to avoid recalculating
trend_cache = {
    'last_updated': None,
    'trend_results': None,
    'papers_by_topic': None,
    'coverage_data': None,
    'visualizations': {}
}

@app.route('/')
def index():
    """Homepage with project overview and quick links"""
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search interface for arXiv papers"""
    results = []
    query = ""
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        category = request.form.get('category', 'cs.AI')
        max_results = int(request.form.get('max_results', 20))
        
        try:
            results = get_arxiv_papers(
                query=query,
                category=category,
                max_results=max_results
            )
            flash(f"Found {len(results)} papers matching your query", "success")
        except Exception as e:
            logger.error(f"Search error: {e}")
            flash(f"Error searching papers: {str(e)}", "danger")
    
    categories = [
        'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.IR',
        'stat.ML', 'quant-ph', 'physics.comp-ph'
    ]
    
    return render_template(
        'search.html',
        results=results,
        query=query,
        categories=categories
    )

@app.route('/trends')
def trends():
    """Display trending topics analysis"""
    # Check if we need to refresh cached data (older than 1 day)
    refresh_needed = (
        not trend_cache['last_updated'] or
        (datetime.now() - trend_cache['last_updated']).days >= 1
    )
    
    if refresh_needed:
        try:
            # Get trending topics data
            logger.info("Refreshing trend data...")
            trend_cache['trend_results'] = identify_emerging_trends(
                categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
                initial_sample_size=100,
                months_to_analyze=3
            )
            
            # Get papers for these topics
            topics = get_emerging_topics(
                categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
                sample_size=100
            )
            trend_cache['papers_by_topic'] = scrape_trending_topics(
                max_results_per_topic=20,
                batch_size=5,
                custom_topics=topics
            )
            
            # Analyze coverage
            trend_cache['coverage_data'] = analyze_topic_coverage(
                trend_cache['papers_by_topic']
            )
            
            # Generate visualizations
            generate_visualizations()
            
            trend_cache['last_updated'] = datetime.now()
            flash("Trend data refreshed successfully", "success")
        except Exception as e:
            logger.error(f"Error refreshing trend data: {e}")
            flash(f"Error refreshing trend data: {str(e)}", "danger")
    
    return render_template(
        'trends.html',
        trend_data=trend_cache['trend_results'],
        coverage_data=trend_cache['coverage_data'],
        visualizations=trend_cache['visualizations'],
        last_updated=trend_cache['last_updated']
    )

@app.route('/papers/<topic_key>')
def papers(topic_key):
    """Display papers for a specific topic"""
    if not trend_cache['papers_by_topic'] or topic_key not in trend_cache['papers_by_topic']:
        flash("Topic not found or trend data not loaded", "warning")
        return redirect(url_for('trends'))
    
    papers = trend_cache['papers_by_topic'][topic_key]
    topic_info = {}
    
    if papers and len(papers) > 0:
        metadata = papers[0].get('metadata', {})
        topic_info = {
            'name': topic_key,
            'description': metadata.get('topic_description', 'No description'),
            'confidence': metadata.get('topic_confidence', 0)
        }
    
    return render_template(
        'papers.html',
        topic_info=topic_info,
        papers=papers
    )

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

def generate_visualizations():
    """Generate visualizations for trend data"""
    if not trend_cache['trend_results'] or not trend_cache['coverage_data']:
        return
    
    try:
        # Create visualizations
        trend_cache['visualizations']['top_trends'] = visualizer.visualize_top_trends(
            trend_cache['trend_results'],
            show=False
        )
        
        trend_cache['visualizations']['wordcloud'] = visualizer.create_topic_wordcloud(
            trend_cache['trend_results'],
            show=False
        )
        
        trend_cache['visualizations']['distribution'] = visualizer.visualize_topic_coverage(
            trend_cache['coverage_data'],
            show=False
        )
        
        trend_cache['visualizations']['by_year'] = visualizer.visualize_topic_papers_by_year(
            trend_cache['coverage_data'],
            show=False
        )
        
        # Convert paths to web URLs
        for key, path in trend_cache['visualizations'].items():
            if path:
                # Extract the filename from the path
                filename = os.path.basename(path)
                trend_cache['visualizations'][key] = url_for('static', filename=f'images/generated/{filename}')
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

if __name__ == '__main__':
    app.run(debug=True)