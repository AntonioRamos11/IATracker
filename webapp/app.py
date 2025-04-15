from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
import os
import sys
import logging
from datetime import datetime
import re
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules with better error handling
modules_imported = True
try:
    # Try to import from Scrapers package
    from Scrapers import get_arxiv_papers, TopicProcessor
    from Scrapers import identify_emerging_trends, get_emerging_topics, scrape_trending_topics, analyze_topic_coverage
    
    # This might be a separate import if not included in __init__.py
    try:
        from Scrapers.visualize_trends import TrendVisualizer
    except ImportError:
        # Create a minimal implementation if not available
        class TrendVisualizer:
            def __init__(self, output_dir=None):
                self.output_dir = output_dir
            def visualize_top_trends(self, *args, **kwargs):
                return None
            def create_topic_wordcloud(self, *args, **kwargs):
                return None
            def visualize_topic_coverage(self, *args, **kwargs):
                return None
            def visualize_topic_papers_by_year(self, *args, **kwargs):
                return None
            
except ImportError as e:
    modules_imported = False
    print(f"Error importing project modules: {e}")
    print("\nTrying alternative import approach...")
    
    # Try individual imports as a fallback
    try:
        sys.path.insert(0, os.path.join(project_root, "Scrapers"))
        from arxiv import get_arxiv_papers
        from topic_processor import TopicProcessor
        from trending_topics import (
            identify_emerging_trends,
            get_emerging_topics,
            scrape_trending_topics,
            analyze_topic_coverage
        )
        try:
            from visualize_trends import TrendVisualizer
        except ImportError:
            class TrendVisualizer:
                def __init__(self, output_dir=None):
                    self.output_dir = output_dir
                def visualize_top_trends(self, *args, **kwargs):
                    return None
                def create_topic_wordcloud(self, *args, **kwargs):
                    return None
                def visualize_topic_coverage(self, *args, **kwargs):
                    return None
                def visualize_topic_papers_by_year(self, *args, **kwargs):
                    return None
        
        modules_imported = True
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("Starting app with limited functionality")
        
        # Define minimal fallback implementations
        def get_arxiv_papers(*args, **kwargs):
            return []
            
        class TopicProcessor:
            def __init__(self):
                pass
        
        def identify_emerging_trends(*args, **kwargs):
            return {"top_trends": []}
            
        def get_emerging_topics(*args, **kwargs):
            return {}
            
        def scrape_trending_topics(*args, **kwargs):
            return {}
            
        def analyze_topic_coverage(*args, **kwargs):
            return {}
            
        class TrendVisualizer:
            def __init__(self, *args, **kwargs):
                pass
            def visualize_top_trends(self, *args, **kwargs):
                return None
            def create_topic_wordcloud(self, *args, **kwargs):
                return None
            def visualize_topic_coverage(self, *args, **kwargs):
                return None
            def visualize_topic_papers_by_year(self, *args, **kwargs):
                return None

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
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'webapp', 'static', 'trend_visualizations')
visualizer = TrendVisualizer(output_dir=static_dir)

# Cache for trend data to avoid recalculating
trend_cache = {
    'last_updated': None,
    'trend_results': None,
    'papers_by_topic': None,
    'coverage_data': None,
    'visualizations': {}
}

# Add this function to load existing visualizations
def load_existing_visualizations():
    """Load existing visualization files from trend_visualizations directory"""
    viz_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "trend_visualizations"
    
    if not viz_dir.exists():
        logger.warning(f"Visualizations directory not found: {viz_dir}")
        return {}
    
    visualizations = {}
    
    # Look for specific visualization types
    papers_by_year = list(viz_dir.glob("papers_by_year_*.png"))
    if papers_by_year:
        latest = max(papers_by_year, key=lambda p: p.stat().st_mtime)
        visualizations['by_year'] = f"/static/trend_visualizations/{latest.name}"
    
    topic_coverage = list(viz_dir.glob("topic_coverage_*.png"))
    if topic_coverage:
        latest = max(topic_coverage, key=lambda p: p.stat().st_mtime)
        visualizations['distribution'] = f"/static/trend_visualizations/{latest.name}"
    
    # Add more visualization types as needed (wordcloud, top_trends)
    
    logger.info(f"Loaded {len(visualizations)} existing visualizations")
    return visualizations

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
    # Initialize visualizations dictionary if it doesn't exist
    if 'visualizations' not in trend_cache:
        trend_cache['visualizations'] = {}
    
    # Load existing visualizations if none present
    if not trend_cache['visualizations']:
        trend_cache['visualizations'] = load_existing_visualizations()
    
    # Initialize the cache if it doesn't exist
    if 'papers_by_topic' not in trend_cache:
        trend_cache['papers_by_topic'] = {}
    
    # Get 'refresh' parameter from URL
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    # Check if we need to refresh cached data
    refresh_needed = (
        refresh or 
        not trend_cache.get('last_updated') or
        (datetime.now() - trend_cache.get('last_updated', datetime.min)).days >= 1 or
        not trend_cache.get('papers_by_topic')
    )
    
    if refresh_needed:
        try:
            # Scan local papers from disk
            local_papers = scan_local_papers()
            
            if local_papers:
                # Use local papers
                trend_cache['papers_by_topic'] = local_papers
                trend_cache['source'] = 'local'
                flash(f"Loaded {sum(len(papers) for papers in local_papers.values())} papers from local storage", "success")
            else:
                # Try to get emerging topics as fallback
                flash("No local papers found. Trying to fetch from arXiv...", "warning")
                topics = get_emerging_topics(categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV"], sample_size=100)
                
                if not topics:
                    # Use sample topics if no real topics found
                    topics = generate_sample_topics()
                    
                # Get papers for these topics
                trend_cache['papers_by_topic'] = scrape_trending_topics(
                    max_results_per_topic=20,
                    batch_size=5,
                    custom_topics=topics
                )
                trend_cache['source'] = 'arxiv'
            
            # Update timestamp
            trend_cache['last_updated'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error refreshing trend data: {e}")
            flash(f"Error loading papers: {str(e)}", "danger")
    
    return render_template(
        'trends.html',
        trend_data=trend_cache.get('trend_results', {}),
        coverage_data=trend_cache.get('coverage_data', {}),
        visualizations=trend_cache.get('visualizations', {}),
        papers_by_topic=trend_cache.get('papers_by_topic', {}),
        last_updated=trend_cache.get('last_updated'),
        source=trend_cache.get('source', 'unknown')
    )

# Helper function to generate sample papers
def _generate_sample_papers():
    """Generate sample paper data for testing"""
    sample_topics = {
        "sample_topic_1": {
            "description": "Sample Topic 1: Large Language Models",
            "category": "cs.CL"
        },
        "sample_topic_2": {
            "description": "Sample Topic 2: Computer Vision",
            "category": "cs.CV"
        }
    }
    
    result = {}
    
    for topic_key, info in sample_topics.items():
        papers = []
        for i in range(5):  # 5 sample papers per topic
            papers.append({
                "title": f"Sample Paper {i+1} for {info['description']}",
                "authors": ["Author One", "Author Two", "Author Three"],
                "summary": "This is a sample paper abstract for testing the web interface. It contains placeholder text that would normally describe the research conducted in this paper.",
                "published": datetime.now(),
                "link": "https://arxiv.org",
                "pdf": "https://arxiv.org/pdf/sample.pdf",
                "categories": [info['category']],
                "metadata": {
                    "topic": topic_key,
                    "topic_description": info['description'],
                    "topic_confidence": 0.85
                }
            })
        result[topic_key] = papers
    
    return result

def generate_sample_topics():
    """Generate sample topics when real topics cannot be found"""
    print("Generating sample topics since no real topics were found")
    
    return {
        "sample_ai_agents": {
            "query": 'ti:"agent" AND cat:cs.AI',
            "category": "cs.AI",
            "description": "AI Agents and Multi-Agent Systems",
            "original_terms": ["agent", "multi-agent", "autonomous"],
            "confidence": 0.85,
            "is_emerging_trend": True
        },
        "sample_language_models": {
            "query": 'ti:"large language model" OR ti:LLM',
            "category": "cs.CL",
            "description": "Large Language Models",
            "original_terms": ["llm", "gpt", "language model"],
            "confidence": 0.95,
            "is_emerging_trend": True
        },
        "sample_diffusion_models": {
            "query": 'ti:"diffusion" AND cat:cs.CV',
            "category": "cs.CV",
            "description": "Diffusion Models",
            "original_terms": ["diffusion", "generative", "image"],
            "confidence": 0.9,
            "is_emerging_trend": True
        },
        "sample_reinforcement_learning": {
            "query": 'ti:"reinforcement learning" AND cat:cs.LG',
            "category": "cs.LG",
            "description": "Reinforcement Learning",
            "original_terms": ["reinforcement", "rl", "policy"],
            "confidence": 0.87,
            "is_emerging_trend": True
        },
        "sample_neural_architecture": {
            "query": 'ti:"neural architecture" AND cat:cs.LG',
            "category": "cs.LG",
            "description": "Neural Architecture Search",
            "original_terms": ["architecture", "nas", "network design"],
            "confidence": 0.82,
            "is_emerging_trend": True
        }
    }

@app.route('/papers/<topic_key>')
def papers(topic_key):
    """Display papers for a specific topic"""
    if not trend_cache['papers_by_topic'] or topic_key not in trend_cache['papers_by_topic']:
        # Add debugging output
        print(f"DEBUG: papers_by_topic keys: {trend_cache['papers_by_topic'].keys() if trend_cache['papers_by_topic'] else 'None'}")
        print(f"DEBUG: Requested topic_key: {topic_key}")
        
        flash("Topic not found or trend data not loaded", "warning")
        return redirect(url_for('trends'))
    
    papers = trend_cache['papers_by_topic'][topic_key]
    print(f"DEBUG: Found {len(papers)} papers for topic {topic_key}")
    
    # Build topic info from the first paper's metadata
    topic_info = {}
    if papers and len(papers) > 0:
        metadata = papers[0].get('metadata', {})
        topic_info = {
            'name': topic_key,
            'description': metadata.get('topic_description', topic_key),
            'confidence': metadata.get('topic_confidence', 0.5)  # Default confidence if not specified
        }
    else:
        topic_info = {
            'name': topic_key,
            'description': 'No description available',
            'confidence': 0.0
        }
    
    return render_template(
        'papers.html',
        topic_info=topic_info,
        papers=papers
    )

@app.route('/papers/<topic>/<filename>')
def serve_paper(topic, filename):
    """Serve a PDF file from the papers directory"""
    papers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
    return send_from_directory(os.path.join(papers_dir, topic), filename)

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/debug')
def debug_info():
    """Show debug information about the application environment"""
    if not request.remote_addr.startswith('127.0.0.1'):
        return "Access denied", 403
        
    debug_data = {
        "python_version": sys.version,
        "python_path": sys.path,
        "project_root": project_root,
        "modules_imported": modules_imported,
        "scrapers_dir_exists": os.path.exists(os.path.join(project_root, "Scrapers")),
        "scrapers_files": os.listdir(os.path.join(project_root, "Scrapers")) if os.path.exists(os.path.join(project_root, "Scrapers")) else [],
        "webapp_dir_exists": os.path.exists(os.path.join(project_root, "webapp")),
        "webapp_files": os.listdir(os.path.join(project_root, "webapp")) if os.path.exists(os.path.join(project_root, "webapp")) else [],
    }
    
    html = "<h1>Debug Information</h1>"
    html += "<pre>"
    for key, value in debug_data.items():
        html += f"{key}: {value}\n"
    html += "</pre>"
    
    return html

@app.route('/debug/papers')
def debug_papers():
    """Debug endpoint to check papers directory"""
    papers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
    
    html = "<h1>Papers Directory Structure</h1>"
    
    if not os.path.exists(papers_dir):
        html += f"<p>Papers directory not found: {papers_dir}</p>"
        return html
    
    html += f"<p>Papers directory: {papers_dir}</p>"
    html += "<ul>"
    
    # Show all subdirectories
    for item in os.listdir(papers_dir):
        item_path = os.path.join(papers_dir, item)
        if os.path.isdir(item_path):
            html += f"<li>📁 {item}"
            # Count PDFs in this directory
            pdfs = [f for f in os.listdir(item_path) if f.lower().endswith('.pdf')]
            html += f" ({len(pdfs)} PDFs)"
            
            if pdfs:
                html += "<ul>"
                for pdf in pdfs[:5]:  # Show first 5 PDFs
                    html += f"<li>📄 {pdf}</li>"
                if len(pdfs) > 5:
                    html += f"<li>... and {len(pdfs)-5} more</li>"
                html += "</ul>"
            
            html += "</li>"
        else:
            html += f"<li>📄 {item}</li>"
    
    html += "</ul>"
    return html

@app.route('/debug/paths')
def debug_paths():
    """Debug endpoint to check file paths"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    paths_to_check = {
        "project_root": project_root,
        "scrapers_dir": os.path.join(project_root, "Scrapers"),
        "trend_cache_dir": os.path.join(project_root, "trend_cache"),
        "webapp_dir": os.path.join(project_root, "webapp"),
    }
    
    html = "<h1>Path Debugging</h1><ul>"
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        html += f"<li>{name}: {path} - {'✅ EXISTS' if exists else '❌ MISSING'}</li>"
        if exists and os.path.isdir(path):
            try:
                files = os.listdir(path)
                html += f"<ul>"
                for file in files:
                    file_path = os.path.join(path, file)
                    is_dir = os.path.isdir(file_path)
                    html += f"<li>{'📁' if is_dir else '📄'} {file}</li>"
                html += f"</ul>"
            except Exception as e:
                html += f"<ul><li>Error listing directory: {str(e)}</li></ul>"
    
    html += "</ul>"
    return html

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

def scan_local_papers():
    """Scan the papers/ directory for local paper files and organize them by topics"""
    papers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
    
    if not os.path.exists(papers_dir):
        print(f"Papers directory not found: {papers_dir}")
        return {}
        
    print(f"Scanning papers directory: {papers_dir}")
    
    papers_by_topic = {}
    
    # Assume each subdirectory is a topic
    for topic_dir in os.listdir(papers_dir):
        topic_path = os.path.join(papers_dir, topic_dir)
        
        if not os.path.isdir(topic_path):
            continue
            
        # Create a clean topic key
        topic_key = f"local_topic_{topic_dir.lower().replace(' ', '_')}"
        papers = []
        
        print(f"Processing topic: {topic_dir}")
        
        # Scan for PDF files
        for file in os.listdir(topic_path):
            if not file.lower().endswith('.pdf'):
                continue
                
            file_path = os.path.join(topic_path, file)
            
            # Extract paper info from filename
            paper_info = extract_paper_info_from_filename(file, topic_dir)
            
            # Add file path
            paper_info['local_path'] = file_path
            paper_info['pdf'] = f"/papers/{topic_dir}/{file}"  # URL for direct download
            
            papers.append(paper_info)
            
        if papers:
            papers_by_topic[topic_key] = papers
            print(f"Found {len(papers)} papers for topic {topic_dir}")
            
    return papers_by_topic

def extract_paper_info_from_filename(filename, topic):
    """Extract paper metadata from filename"""
    # Remove .pdf extension
    name = os.path.splitext(filename)[0]
    
    # Try to extract year (assuming format like "Paper Title (2023).pdf")
    year_match = re.search(r'\((\d{4})\)$', name)
    year = year_match.group(1) if year_match else None
    
    # Remove the year part from the title
    title = re.sub(r'\(\d{4}\)$', '', name).strip()
    
    # Create a paper object
    return {
        'title': title,
        'authors': ['Unknown'],  # Metadata not available from filename
        'summary': f"A paper on {topic}. Detailed abstract not available for local papers.",
        'published': datetime.strptime(f"{year or '2023'}-01-01", "%Y-%m-%d") if year else None,
        'link': None,
        'categories': [topic],
        'metadata': {
            'topic': topic,
            'topic_description': topic,
            'topic_confidence': 1.0
        }
    }

def load_processed_papers():
    """Load processed papers and their highlights from the Database directory"""
    database_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Database")
    highlights_dir = os.path.join(database_dir, "paper_highlights")
    processed_dir = os.path.join(database_dir, "processed_pdfs")
    
    if not os.path.exists(database_dir):
        logger.warning(f"Database directory not found: {database_dir}")
        return []
    
    papers = []
    
    # First scan the highlights directory
    if os.path.exists(highlights_dir):
        import json
        logger.info(f"Scanning highlights directory: {highlights_dir}")
        
        for filename in os.listdir(highlights_dir):
            if filename.endswith('_highlights.json'):
                try:
                    # Extract the content hash from filename
                    content_hash = filename.replace('_highlights.json', '')
                    highlight_path = os.path.join(highlights_dir, filename)
                    
                    # Load the JSON data
                    with open(highlight_path, 'r') as f:
                        highlight_data = json.load(f)
                    
                    # Check if processed PDF exists
                    pdf_exists = False
                    pdf_path = None
                    if processed_dir and os.path.exists(processed_dir):
                        pdf_path = os.path.join(processed_dir, f"{content_hash}.pdf")
                        pdf_exists = os.path.exists(pdf_path)
                    
                    # Extract metadata
                    metadata = highlight_data.get('metadata', {})
                    main_info = highlight_data.get('main_staff', {})
                    
                    # Create structured highlights from full_response if available
                    structured_highlights = []
                    full_response = highlight_data.get('full_response', '')
                    
                    if '### Structured Highlights for the Research Paper' in full_response:
                        sections = full_response.split('#### ')
                        for section in sections[1:]:  # Skip the first element
                            if ':' in section:
                                section_title = section.split('\n')[0].strip()
                                section_content = '\n'.join(section.split('\n')[1:]).strip()
                                structured_highlights.append({
                                    'title': section_title,
                                    'content': section_content
                                })
                    
                    # Create paper object
                    paper = {
                        'id': content_hash,
                        'title': metadata.get('title', 'Unknown Title').replace('_', ' '),
                        'authors': main_info.get('authors', []),
                        'abstract': main_info.get('abstract', ''),
                        'topics': highlight_data.get('topics', []),
                        'full_response': highlight_data.get('full_response', ''),
                        'structured_highlights': structured_highlights,
                        'sections': main_info.get('sections', []),
                        'published': metadata.get('published', ''),
                        'pages': metadata.get('pages', 0),
                        'source_path': metadata.get('source_path', '')
                    }
                    
                    # Add processed PDF path if exists
                    if pdf_exists:
                        paper['processed_pdf'] = f"/database/processed_pdf/{content_hash}.pdf"
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error loading highlight file {filename}: {e}")
    
    logger.info(f"Loaded {len(papers)} processed papers with highlights")
    return papers

def clean_paper_title(filename):
    """Convert filename to readable title"""
    # Remove extension if present
    base = os.path.splitext(filename)[0]
    
    # Replace underscores and hyphens with spaces
    title = base.replace('_', ' ').replace('-', ' ')
    
    # Capitalize words
    title = ' '.join(word.capitalize() for word in title.split())
    
    return title

@app.route('/database')
def database():
    """Display processed papers with highlights"""
    papers = load_processed_papers()
    return render_template('database.html', papers=papers)

@app.route('/database/processed_pdf/<filename>')
def serve_processed_pdf(filename):
    """Serve a processed PDF file from the database"""
    database_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Database")
    return send_from_directory(os.path.join(database_dir, "processed_pdfs"), filename)

@app.route('/paper/<paper_id>')
def paper_highlight(paper_id):
    """Display detailed highlights for a specific paper"""
    database_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Database")
    highlights_dir = os.path.join(database_dir, "paper_highlights")
    
    highlight_file = os.path.join(highlights_dir, f"{paper_id}_highlights.json")
    
    if not os.path.exists(highlight_file):
        flash("Highlights not found for this paper", "warning")
        return redirect(url_for('database'))
    
    try:
        import json
        with open(highlight_file, 'r') as f:
            highlight_data = json.load(f)
            
        # Extract metadata
        metadata = highlight_data.get('metadata', {})
        main_info = highlight_data.get('main_staff', {})
        
        # Create structured highlights from full_response if available
        structured_highlights = []
        full_response = highlight_data.get('full_response', '')
        
        if '### Structured Highlights for the Research Paper' in full_response:
            sections = full_response.split('#### ')
            for section in sections[1:]:  # Skip the first element
                if ':' in section:
                    section_title = section.split('\n')[0].strip()
                    section_content = '\n'.join(section.split('\n')[1:]).strip()
                    structured_highlights.append({
                        'title': section_title,
                        'content': section_content
                    })
        
        # Create paper object
        paper = {
            'id': paper_id,
            'title': metadata.get('title', 'Unknown Title').replace('_', ' '),
            'authors': main_info.get('authors', []),
            'abstract': main_info.get('abstract', ''),
            'topics': highlight_data.get('topics', []),
            'full_response': highlight_data.get('full_response', ''),
            'structured_highlights': structured_highlights,
            'sections': main_info.get('sections', []),
            'published': metadata.get('published', ''),
            'pages': metadata.get('pages', 0),
            'source_path': metadata.get('source_path', '')
        }
        
        # Check if processed PDF exists
        pdf_path = os.path.join(database_dir, "processed_pdfs", f"{paper_id}.pdf")
        if os.path.exists(pdf_path):
            paper['processed_pdf'] = f"/database/processed_pdf/{paper_id}.pdf"
        
    except Exception as e:
        logger.error(f"Error loading highlights for {paper_id}: {e}")
        flash(f"Error loading paper data: {str(e)}", "danger")
        return redirect(url_for('database'))
    
    return render_template('paper_highlight.html', paper=paper)

if __name__ == '__main__':
    app.run(debug=True)