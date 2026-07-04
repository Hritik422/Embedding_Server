import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI(title="Vector Embedding API")

# FIX: Set the cache directory explicitly to Vercel's writeable '/tmp' directory
# This stops permission crashes on serverless architectures
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="/tmp/fastembed_cache"
)

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(payload: EmbeddingRequest):
    embeddings_generator = model.embed([payload.text])
    embedding_list = list(embeddings_generator)[0].tolist()
    
    return {"embedding": embedding_list}