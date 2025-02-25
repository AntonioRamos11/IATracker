import psycopg2

DB_CONFIG = {
    "dbname": "ai_papers",
    "user": "postgres",
    "password": "Chappie1101",
    "host": "localhost",
    "port": 5432  
}

# Connect to the database using the configuration above
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# If you haven't already enabled the pgvector extension, run this:
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# Now, execute the command to create the table
create_table_query = """
CREATE TABLE paper_vectors (
    id SERIAL PRIMARY KEY,
    title TEXT,
    metadata JSONB,
    embedding vector(768)  -- 768-d embeddings (OpenAI's model size)
);
"""
cur.execute(create_table_query)
conn.commit()

cur.close()
conn.close()
