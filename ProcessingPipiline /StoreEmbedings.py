import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch
from psycopg2 import sql, errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Base exception for vector storage errors"""

class Configuration:
    """Centralized configuration"""
    MODEL_NAME = "all-mpnet-base-v2"  # Update to a model with 768 dimensions
    DB_CONFIG = {
        "dbname": "ai_papers",
        "user": "postgres",
        "password": "Chappie1101",
        "host": "localhost",
        "port": 5432
    }
    BATCH_SIZE = 100
    MAX_RETRIES = 3
    EMBEDDING_DIM = 768  # Must match model output dimension

class VectorStore:
    def __init__(self):
        self.model = self._load_model()
        self.pool = self._create_connection_pool()

    def _load_model(self) -> SentenceTransformer:
        """Load model with validation"""
        try:
            model = SentenceTransformer(Configuration.MODEL_NAME)
            assert model.get_sentence_embedding_dimension() == Configuration.EMBEDDING_DIM
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise VectorStoreError("Failed to initialize model") from e

    def _create_connection_pool(self) -> ThreadedConnectionPool:
        """Create connection pool with validation"""
        try:
            pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                **Configuration.DB_CONFIG
            )
            # Test connection
            with pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return pool
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise VectorStoreError("Database connection failed") from e

    @retry(
        stop=stop_after_attempt(Configuration.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((errors.OperationalError, errors.InterfaceError))
    )
    def _execute_batch(self, query: sql.Composed, data: List[tuple]) -> None:
        """Execute batch insert with retry logic"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                execute_batch(cursor, query, data, page_size=Configuration.BATCH_SIZE)
            conn.commit()
        except errors.UniqueViolation:
            logger.warning("Duplicate entries detected, skipping")
            conn.rollback()
        except Exception as e:
            conn.rollback()
            logger.error(f"Batch insert failed: {str(e)}")
            raise
        finally:
            self.pool.putconn(conn)

    def _convert_embedding(self, embedding: np.ndarray) -> str:
        """Convert numpy array to PostgreSQL vector string"""
        return f"[{','.join(map(str, embedding))}]"

    def store_embeddings(self, documents: List[Dict[str, Any]]) -> None:
        """Store batch of documents with embeddings"""
        if not documents:
            return

        # Generate embeddings in batch
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Prepare data for batch insert
        data = []
        for doc, embedding in zip(documents, embeddings):
            try:
                data.append((
                    doc["metadata"]["title"],
                    json.dumps(doc["metadata"]),
                    self._convert_embedding(embedding)
                ))
            except KeyError as e:
                logger.error(f"Invalid document format: {str(e)}")
                continue

        # Build parameterized query
        query = sql.SQL("""
            INSERT INTO paper_vectors (title, metadata, embedding)
            VALUES (%s, %s, %s::vector({dim}))
            ON CONFLICT DO NOTHING
        """).format(dim=sql.SQL(str(Configuration.EMBEDDING_DIM)))

        try:
            self._execute_batch(query, data)
            logger.info(f"Successfully stored {len(data)} embeddings")
        except Exception as e:
            logger.error(f"Failed to store batch: {str(e)}")
            raise VectorStoreError("Batch storage failed") from e

    def process_directory(self, processed_dir: Path) -> None:
        """Process all JSON files in directory"""
        batch = []
        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    self._validate_document(data)
                    batch.append(data)

                if len(batch) >= Configuration.BATCH_SIZE:
                    self.store_embeddings(batch)
                    batch = []

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error processing {json_file.name}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error with {json_file.name}: {str(e)}")
                continue

        # Process remaining documents
        if batch:
            self.store_embeddings(batch)

    def _validate_document(self, doc: Dict) -> None:
        """Validate document structure"""
        required = {"text", "metadata"}
        if not required.issubset(doc.keys()):
            raise ValueError("Missing required document fields")
        if "title" not in doc["metadata"]:
            raise ValueError("Missing title in metadata")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "pool"):
            self.pool.closeall()

if __name__ == "__main__":
    try:
        store = VectorStore()
        processed_dir = Path("./Database/processed_pdfs/")
        
        if not processed_dir.exists():
            raise FileNotFoundError(f"Directory {processed_dir} not found")

        store.process_directory(processed_dir)
        logger.info("Embedding storage completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}", exc_info=True)
        raise SystemExit(1) from e