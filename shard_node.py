import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# --- State ---
SHARD_ID = os.getenv("SHARD_ID", "shard_0")
TOPIC = os.getenv("SHARD_TOPIC", "General")
print(f"🐝 [Shard {SHARD_ID}] Initializing... Topic: {TOPIC}")

# Load Embedding Model (Shared or Local)
# In production, shards might just do vector math if embeddings are pre-computed,
# but for standalone robustness, we load the model.
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Mock Knowledge Base (In prod, load from Disk/DB)
KNOWLEDGE_BASE = []

def seed_knowledge():
    print(f"🌱 Seeding Knowledge for {TOPIC}...")
    topics_data = {
        "Science": [
            "The speed of light is 299,792,458 m/s.",
            "Mitochondria is the powerhouse of the cell.",
            "Water boils at 100 degrees Celsius at sea level."
        ],
        "History": [
            "The Roman Empire fell in 476 AD.",
            "World War II ended in 1945.",
            "The Great Wall of China was built over centuries."
        ],
        "Coding": [
            "Python uses indentation for blocks.",
            "React uses a virtual DOM for performance.",
            "Docker containers share the host OS kernel."
        ]
    }
    
    facts = topics_data.get(TOPIC, ["Generic fact 1", "Generic fact 2"])
    
    for i, text in enumerate(facts):
        vec = embedder.encode(text)
        KNOWLEDGE_BASE.append({
            "id": f"{SHARD_ID}_{i}",
            "text": text,
            "vector": vec,
            "metadata": {"centrality": 0.9, "recency": 1.0, "weight": 1.0}
        })
    print(f"✅ Loaded {len(KNOWLEDGE_BASE)} facts.")

@app.on_event("startup")
async def startup_event():
    seed_knowledge()

# --- API ---

class RetrievalRequest(BaseModel):
    query_text: str
    query_vector: List[float] = None # Optional, if router computed it

class RetrievalResponse(BaseModel):
    shard_id: str
    best_text: str
    score: float

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest):
    # 1. Get Query Vector
    if req.query_vector:
        q_vec = np.array(req.query_vector)
    else:
        q_vec = embedder.encode(req.query_text)
        
    # 2. Vector Search (Dot Product)
    # Stack all vectors
    db_vecs = np.array([item["vector"] for item in KNOWLEDGE_BASE])
    
    # Cosine Sim
    norm_q = np.linalg.norm(q_vec)
    norm_db = np.linalg.norm(db_vecs, axis=1)
    dot = np.dot(db_vecs, q_vec)
    sims = dot / (norm_q * norm_db + 1e-9)
    
    # 3. Apply Metadata (P = S * C * R * W)
    best_score = -1.0
    best_idx = 0
    
    for i, sim in enumerate(sims):
        meta = KNOWLEDGE_BASE[i]["metadata"]
        p_score = sim * meta["centrality"] * meta["recency"] * meta["weight"]
        
        if p_score > best_score:
            best_score = p_score
            best_idx = i
            
    return RetrievalResponse(
        shard_id=SHARD_ID,
        best_text=KNOWLEDGE_BASE[best_idx]["text"],
        score=float(best_score)
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
