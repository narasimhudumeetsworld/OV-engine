import os
import aiohttp
import asyncio
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="OpenVinayaka Hive Router", version="3.5")

# Configuration
SHARD_URLS = os.getenv("SHARD_URLS", "http://localhost:8001,http://localhost:8002,http://localhost:8003").split(",")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    model: str

async def query_shard(session, url, query_text):
    try:
        async with session.post(f"{url}/retrieve", json={"query_text": query_text}) as resp:
            return await resp.json()
    except Exception as e:
        print(f"⚠️ Shard {url} failed: {e}")
        return None

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    user_query = request.messages[-1].content
    print(f"👑 [Router] Received: '{user_query}'")
    
    # 1. Broadcast to Shards (The Hive Mind)
    start_t = time.time()
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [query_shard(session, url, user_query) for url in SHARD_URLS]
        results = await asyncio.gather(*tasks)
    
    # 2. Consensus (Filter failures)
    valid_results = [r for r in results if r]
    
    if not valid_results:
        best_truth = "No knowledge found."
    else:
        # Find highest P-Score
        winner = max(valid_results, key=lambda x: x["score"])
        best_truth = winner["best_text"]
        print(f"✅ [Consensus] Winner: {winner['shard_id']} (Score: {winner['score']:.4f})")
    
    # 3. Generate (Simulated LLM for this distributed test)
    # In a full deployment, we'd load the OVModelManager here.
    # For this Docker test, we focus on the Network Topology.
    final_response = f"[OV-Memory Context: {best_truth}] \n\nBased on this truth, here is the answer..."
    
    return ChatCompletionResponse(
        id=f"ov-{int(time.time())}",
        model=request.model,
        choices=[{
            "message": {"role": "assistant", "content": final_response},
            "finish_reason": "stop"
        }]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
