import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from typing import Dict, Any, Optional

app = FastAPI(title="Vector Embedding API")

model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="/tmp/fastembed_cache"
)

class EmbeddingRequest(BaseModel):
    text: str
    # Dict[str, Any] allows context to accept any valid JSON object/dictionary
    context: Optional[Any] = None  

@app.post("/embed")
def get_embedding(payload: EmbeddingRequest):
    embeddings_generator = model.embed([payload.text])
    embedding_list = list(embeddings_generator)[0].tolist()
    
    return {
        "embedding": embedding_list, 
        "context": payload.context
    }