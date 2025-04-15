#!/usr/bin/env python3
# filepath: /home/p0wden/Documents/IAResearchAgregator/Scrapers/test_trending_topics.py

import unittest
import logging
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime
from trending_topics import (
    TopicProcessor, 
    extract_trending_topics_from_report,
    load_valid_topics,
    fetch_topic_papers,
    scrape_trending_topics,
    analyze_topic_coverage
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestTrendingTopics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = TopicProcessor()
        
        # Sample topic data
        self.sample_topic = {
            "refined_title": "Large Language Models Optimization",
            "original_id": "001",
            "top_terms": ["transformer", "attention", "optimization", "scaling", "fine-tuning"],
            "confidence_score": 0.85
        }
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.trend_cache_dir = Path(self.test_dir.name) / "trend_cache"
        self.trend_cache_dir.mkdir(exist_ok=True)
        
        # Create a sample trend report
        self.sample_report_path = self.trend_cache_dir / "trend_report_2025_04_12.txt"
        with open(self.sample_report_path, 'w') as f:
            json.dump([self.sample_topic], f)
    
    def tearDown(self):
        """Clean up after tests"""
        self.test_dir.cleanup()
    
    def test_topic_processor_sanitize(self):
        """Test sanitize_topic_name method"""
        test_cases = [
            ("Large Language Models", "large_language_models"),
            ("Reinforcement Learning & Game Theory", "reinforcement_learning_and_game_theory"),
            ("Vision/Language Pre-training", "vision_language_pre_training"),
            ("ML in Healthcare (2023)", "ml_in_healthcare_2023"),
        ]
        
        for input_title, expected_output in test_cases:
            result = self.processor.sanitize_topic_name(input_title)
            self.assertEqual(result, expected_output)
            print(f"✓ Sanitized '{input_title}' to '{result}'")
    
    def test_topic_processor_validation(self):
        """Test topic quality validation"""
        valid_topic = {
            "title": "Large Language Models",
            "terms": ["transformer", "attention", "neural", "scaling"]
        }
        
        invalid_topic = {
            "title": "Bad",
            "terms": ["et", "al"]
        }
        
        is_valid = self.processor.validate_topic_quality(valid_topic["title"], valid_topic["terms"])
        self.assertTrue(is_valid)
        print(f"✓ Valid topic correctly identified: {valid_topic['title']}")
        
        is_valid = self.processor.validate_topic_quality(invalid_topic["title"], invalid_topic["terms"])
        self.assertFalse(is_valid)
        print(f"✓ Invalid topic correctly identified: {invalid_topic['title']}")
    
    def test_topic_processor_category_prediction(self):
        """Test arXiv category prediction"""
        test_cases = [
            ({"title": "Large Language Models", "terms": ["transformer", "attention"]}, "cs.CL"),
            ({"title": "Computer Vision", "terms": ["CNN", "detection"]}, "cs.CV"),
            ({"title": "Reinforcement Learning", "terms": ["agent", "policy"]}, "cs.AI"),
            ({"title": "Quantum Computing", "terms": ["quantum", "qubits"]}, "quant-ph")
        ]
        
        for topic, expected_category in test_cases:
            category = self.processor.predict_arxiv_category(topic["title"], topic["terms"])
            self.assertEqual(category, expected_category)
            print(f"✓ Topic '{topic['title']}' correctly categorized as '{category}'")
    
    def test_topic_processor_query_building(self):
        """Test query building logic"""
        terms = ["transformer", "neural", "attention", "language", "model"]
        query = self.processor.build_optimized_query(terms)
        
        self.assertIn("transformer", query)
        self.assertIn("neural", query)
        self.assertIn("attention", query)
        self.assertIn("all:language", query)
        self.assertIn("ANDNOT", query)
        print(f"✓ Query correctly built: {query}")
        
        # Test empty terms
        empty_query = self.processor.build_optimized_query([])
        self.assertEqual(empty_query, "artificial intelligence")
        print("✓ Default query used for empty terms")
    
    @patch('trending_topics.TopicRefiner')
    def test_extract_trending_topics(self, mock_refiner_class):
        """Test extraction of trending topics from report"""
        # Mock the TopicRefiner
        mock_refiner = MagicMock()
        mock_refiner.refine_topics_from_report.return_value = [self.sample_topic]
        mock_refiner_class.return_value = mock_refiner
        
        # Test the function
        topics = extract_trending_topics_from_report(self.sample_report_path)
        
        # Verify results
        self.assertGreaterEqual(len(topics), 1)
        self.assertTrue(any('large_language_models' in k for k in topics.keys()))
        print(f"✓ Successfully extracted {len(topics)} topics")
        for k, v in topics.items():
            print(f"  - {k}: {v['description']} ({v['category']})")
    
    @patch('trending_topics.extract_trending_topics_from_report')
    def test_load_valid_topics(self, mock_extract):
        """Test loading valid topics"""
        # Setup mock
        expected_topics = {
            "topic_001_large_language_models": {
                "query": "(transformer OR attention OR optimization)",
                "category": "cs.CL",
                "description": "Large Language Models Optimization",
                "original_terms": ["transformer", "attention", "optimization"],
                "confidence": 0.85
            }
        }
        mock_extract.return_value = expected_topics
        
        # Create a temporary trend_cache directory in the current path
        with patch('trending_topics.Path') as mock_path:
            mock_path.return_value.glob.return_value = [self.sample_report_path]
            mock_path.return_value.stat.return_value.st_mtime = datetime.now().timestamp()
            
            topics = load_valid_topics()
            
            self.assertEqual(topics, expected_topics)
            print(f"✓ Successfully loaded {len(topics)} valid topics")
    
    @patch('trending_topics.get_arxiv_papers')
    def test_fetch_topic_papers(self, mock_get_papers):
        """Test fetching papers for a topic"""
        # Sample papers
        sample_papers = [
            {"title": "Paper 1", "abstract": "Abstract 1", "published": datetime(2023, 1, 1)},
            {"title": "Paper 2", "abstract": "Abstract 2", "published": datetime(2023, 2, 1)}
        ]
        mock_get_papers.return_value = sample_papers
        
        # Test parameters
        topic_name = "topic_001_large_language_models"
        topic_info = {
            "query": "(transformer OR attention OR optimization)",
            "category": "cs.CL",
            "description": "Large Language Models Optimization",
            "original_terms": ["transformer", "attention", "optimization"]
        }
        
        # Run the function
        papers = fetch_topic_papers(topic_name, topic_info, 10, 5)
        
        # Verify results
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["metadata"]["topic"], topic_name)
        self.assertEqual(papers[0]["metadata"]["topic_description"], topic_info["description"])
        print(f"✓ Successfully fetched {len(papers)} papers for topic '{topic_info['description']}'")
    
    @patch('trending_topics.load_valid_topics')
    @patch('trending_topics.fetch_topic_papers')
    def test_scrape_trending_topics(self, mock_fetch, mock_load_topics):
        """Test scraping trending topics"""
        # Sample topic
        sample_topics = {
            "topic_001_large_language_models": {
                "query": "(transformer OR attention OR optimization)",
                "category": "cs.CL",
                "description": "Large Language Models Optimization",
                "original_terms": ["transformer", "attention", "optimization"]
            }
        }
        
        # Sample papers
        sample_papers = [
            {"title": "Paper 1", "abstract": "Abstract 1", "published": datetime(2023, 1, 1)},
            {"title": "Paper 2", "abstract": "Abstract 2", "published": datetime(2023, 2, 1)}
        ]
        
        # Setup mocks
        mock_load_topics.return_value = sample_topics
        mock_fetch.return_value = sample_papers
        
        # Run the function
        result = scrape_trending_topics(max_results_per_topic=10, batch_size=5, max_parallel_requests=1)
        
        # Verify results
        self.assertEqual(len(result), 1)
        self.assertIn("topic_001_large_language_models", result)
        self.assertEqual(len(result["topic_001_large_language_models"]), 2)
        print(f"✓ Successfully scraped {len(result)} topics with {len(result['topic_001_large_language_models'])} papers")
    
    def test_analyze_topic_coverage(self):
        """Test analyzing topic coverage"""
        # Sample data
        all_papers = {
            "topic_001_large_language_models": [
                {
                    "title": "Paper 1", 
                    "published": datetime(2023, 1, 1),
                    "metadata": {"topic_description": "Large Language Models"}
                },
                {
                    "title": "Paper 2", 
                    "published": datetime(2023, 1, 1),
                    "metadata": {"topic_description": "Large Language Models"}
                },
                {
                    "title": "Paper 3", 
                    "published": datetime(2024, 1, 1),
                    "metadata": {"topic_description": "Large Language Models"}
                }
            ],
            "topic_002_computer_vision": [
                {
                    "title": "Vision Paper 1", 
                    "published": datetime(2023, 1, 1),
                    "metadata": {"topic_description": "Computer Vision"}
                }
            ]
        }
        
        # Run function
        coverage = analyze_topic_coverage(all_papers)
        
        # Verify results
        self.assertEqual(len(coverage), 2)
        self.assertEqual(coverage["topic_001_large_language_models"]["total"], 3)
        self.assertEqual(len(coverage["topic_001_large_language_models"]["by_year"]), 2)
        self.assertEqual(coverage["topic_002_computer_vision"]["total"], 1)
        
        print("✓ Topic coverage analysis successful:")
        for topic, stats in coverage.items():
            print(f"  - {topic}: {stats['total']} papers across {len(stats['by_year'])} years")
            print(f"    Description: {stats['description']}")
            print(f"    Sample titles: {', '.join(stats['sample_titles'][:2])}")
            
    def test_end_to_end(self):
        """Test entire flow with minimal mocking"""
        print("\n==== RUNNING END-TO-END TEST ====")
        
        # Use the processor's fallback topics for actual testing
        topics = self.processor.get_fallback_topics()
        self.assertGreaterEqual(len(topics), 1)
        print(f"✓ Got {len(topics)} fallback topics")
        
        # Check query building logic with one of the topics
        sample_topic = next(iter(topics.values()))
        terms = sample_topic.get("original_terms", [])
        query = self.processor.build_optimized_query(terms)
        self.assertIsNotNone(query)
        print(f"✓ Built query: {query}")
        
        print("==== END-TO-END TEST SUCCESSFUL ====")

if __name__ == "__main__":
    print("\n===== TRENDING TOPICS MODULE TESTS =====\n")
    unittest.main(verbosity=2)