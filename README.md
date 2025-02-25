# IATracker
IATracker is a research aggregator and analysis tool designed to fetch, process, and analyze research papers from various sources. It uses advanced natural language processing techniques to identify emerging trends and topics in the field of artificial intelligence.

## Features

- Fetch research papers from arXiv and other sources
- Store and manage PDFs with content-based hashing
- Generate embeddings for research papers using Sentence Transformers
- Store embeddings in a PostgreSQL database
- Analyze trends and identify emerging topics using BERTopic
- Visualize topic distributions and trends over time

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher
- Git
- Make
- GCC

### Setup

1. Clone the repository:

```sh
git clone https://github.com/yourusername/IATracker.git
cd IATracker