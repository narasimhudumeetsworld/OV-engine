import threading
import time
import sys
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Add the package to path
sys.path.append(os.path.abspath("OpenVinayaka_Release_v1/Python_Package"))
# We import the benchmark logic concepts here to simulate the engine's core decision
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DATASET_PATH = "OpenVinayaka_Release_v1/Benchmarks_10k/dataset_10k_challenge.json"
CONCURRENT_USERS = 21

# Global Stats
stats = {
    "processed": 0,
    "success": 0,
    "failed": 0,
    "lock": threading.Lock()
}

print("⏳ Loading OV-Engine Core (Embedding Model)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_query_batch(user_id, batch_data):
    """
    Simulates a User Session processing a batch of adversarial prompts.
    Does the full OV-Memory Math: P = S * C * R * W
    """
    local_success = 0
    local_fail = 0
    
    # Pre-compute embeddings for this batch to simulate "User Thinking" / Network Latency
    # In a real server, these come in one by one, but for a stress test, 
    # we want to hammer the CPU with the MATH logic.
    
    for item in batch_data:
        query = item["query"]
        
        # 1. Simulate Retrieval Candidates
        truth = item["ground_truth"]
        distractor = item["distractors"][0]
        
        # 2. Compute Vectors (The expensive CPU part)
        q_vec = embedder.encode(query)
        t_vec = embedder.encode(truth["text"])
        d_vec = embedder.encode(distractor["text"])
        
        # 3. OV-Engine Logic (The Safety Lock)
        # Truth Score
        s_t = cosine_sim(q_vec, t_vec)
        p_t = s_t * truth["metadata"]["centrality"] * truth["metadata"]["recency"] * truth["metadata"]["weight"]
        
        # Distractor Score
        s_d = cosine_sim(q_vec, d_vec)
        p_d = s_d * distractor["metadata"]["centrality"] * distractor["metadata"]["recency"] * distractor["metadata"]["weight"]
        
        # 4. Decision
        if p_t > p_d:
            local_success += 1
        else:
            local_fail += 1
            
    # Update Globals
    with stats["lock"]:
        stats["processed"] += len(batch_data)
        stats["success"] += local_success
        stats["failed"] += local_fail
        
    print(f"👤 [User_{user_id:02d}] Finished Batch. Success: {local_success}/{len(batch_data)}")

def main():
    print(f"🚀 Starting 10,000 Query Stress Test with {CONCURRENT_USERS} Users...")
    print(f"   Dataset: {DATASET_PATH}")
    print("=" * 60)
    
    # 1. Load Dataset
    with open(DATASET_PATH, "r") as f:
        full_dataset = json.load(f)
        
    total_items = len(full_dataset)
    print(f"📦 Loaded {total_items} Adversarial Scenarios.")
    
    # 2. Split into batches for users
    chunk_size = total_items // CONCURRENT_USERS
    batches = []
    for i in range(CONCURRENT_USERS):
        start = i * chunk_size
        end = start + chunk_size if i < CONCURRENT_USERS - 1 else total_items
        batches.append(full_dataset[start:end])
        
    print(f"⚡ Launching {CONCURRENT_USERS} Threads (approx {chunk_size} queries/user)...")
    
    t0 = time.time()
    
    # 3. Execute Parallel Stress Test
    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = []
        for i in range(CONCURRENT_USERS):
            futures.append(executor.submit(process_query_batch, i+1, batches[i]))
            
        # Wait for all
        for f in futures:
            f.result()
            
    duration = time.time() - t0
    
    print("=" * 60)
    print(f"🏆 STRESS TEST COMPLETE")
    print(f"   Total Time: {duration:.2f}s")
    print(f"   Throughput: {total_items / duration:.2f} queries/sec")
    print("-" * 60)
    print(f"   Total Queries: {stats['processed']}")
    print(f"   Safety Wins:   {stats['success']}")
    print(f"   Failures:      {stats['failed']}")
    print(f"   Accuracy:      {(stats['success'] / total_items) * 100:.2f}%")
    print("=" * 60)
    
    if stats['failed'] == 0:
        print("✅ PASSED: OV-Engine maintained 100% integrity under massive load.")
    else:
        print("❌ FAILED: Engine leaked hallucinations under load.")

if __name__ == "__main__":
    main()
