from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI(title="Vector Embedding API")

# Initializes the model once during instance startup
model = TextEmbedding()

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(payload: EmbeddingRequest):
    embeddings_generator = model.embed([payload.text])
    embedding_list = list(embeddings_generator)[0].tolist()
    
    return {"embedding": embedding_list}
