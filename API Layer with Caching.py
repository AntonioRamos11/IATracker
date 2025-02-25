from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

app = FastAPI()

@app.get("/papers/search")
@cache(expire=300)
async def search_papers(query: str, k: int = 5):
    query_vector = vectorizer.get_embeddings(query)
    results = vector_index.search(query_vector, k)
    return db.get_metadata(results.ids)