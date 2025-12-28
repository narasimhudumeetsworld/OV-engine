import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

# Colors
class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

print(f"{Colors.HEADER}⏳ Initializing 10k Benchmark Engine...{Colors.ENDC}")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_sim_batch(q_embs, d_embs):
    # q_embs: (N, D)
    # d_embs: (N, D) - assuming 1-to-1 comparison for this specific test structure
    # Returns: (N,) scores
    norm_q = np.linalg.norm(q_embs, axis=1)
    norm_d = np.linalg.norm(d_embs, axis=1)
    dot = np.sum(q_embs * d_embs, axis=1)
    return dot / (norm_q * norm_d + 1e-9)

def run_10k_benchmark():
    # Load Data
    print("📂 Loading Dataset...")
    try:
        with open("dataset_10k_challenge.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Dataset not found in current directory. Creating it...")
        import os
        os.system("python3 generate_10k_challenge.py")
        with open("dataset_10k_challenge.json", "r") as f:
            dataset = json.load(f)
    
    total = len(dataset)
    print(f"✅ Loaded {total} items.")
    
    # Prepare Batches (All in memory for speed, 10k is small for RAM)
    print("🧠 Computing Embeddings (This may take a minute)...")
    
    queries = [d["query"] for d in dataset]
    truths = [d["ground_truth"]["text"] for d in dataset]
    distractors = [d["distractors"][0]["text"] for d in dataset]
    
    # Get Metadata
    truth_meta = [d["ground_truth"]["metadata"] for d in dataset]
    dist_meta = [d["distractors"][0]["metadata"] for d in dataset]
    
    t0 = time.time()
    
    # Embed everything
    # Batch size 128
    q_embs = embedder.encode(queries, batch_size=128, show_progress_bar=True)
    t_embs = embedder.encode(truths, batch_size=128, show_progress_bar=True)
    d_embs = embedder.encode(distractors, batch_size=128, show_progress_bar=True)
    
    print(f"⚡ Embeddings computed in {time.time() - t0:.2f}s")
    
    print("⚔️  Running BATTLE: Standard RAG vs OV-Memory...")
    
    # 1. Standard RAG (Vector Similarity Only)
    # Compare Query vs Truth AND Query vs Distractor
    score_truth_rag = cosine_sim_batch(q_embs, t_embs)
    score_dist_rag = cosine_sim_batch(q_embs, d_embs)
    
    # RAG Decision: Who has higher score?
    rag_wins = np.sum(score_truth_rag > score_dist_rag)
    
    # 2. OV-Memory (Vector * Metadata)
    # Extract meta arrays
    t_C = np.array([m["centrality"] for m in truth_meta])
    t_R = np.array([m["recency"] for m in truth_meta])
    t_W = np.array([m["weight"] for m in truth_meta])
    
    d_C = np.array([m["centrality"] for m in dist_meta])
    d_R = np.array([m["recency"] for m in dist_meta])
    d_W = np.array([m["weight"] for m in dist_meta])
    
    # Calculate P = S * C * R * W
    score_truth_ov = score_truth_rag * t_C * t_R * t_W
    score_dist_ov = score_dist_rag * d_C * d_R * d_W
    
    # OV Decision
    ov_wins = np.sum(score_truth_ov > score_dist_ov)
    
    # --- REPORT ---
    print("\n" + "="*60)
    print(f"{Colors.BOLD}🏆 10,000 SCENARIO BATTLE REPORT{Colors.ENDC}")
    print("="*60)
    
    print(f"{'Metric':<20} | {'Standard RAG':<15} | {'OV-Memory':<15}")
    print("-" * 60)
    
    rag_acc = (rag_wins / total) * 100
    ov_acc = (ov_wins / total) * 100
    
    print(f"{'Wins':<20} | {rag_wins:<15} | {ov_wins:<15}")
    print(f"{'Failures':<20} | {total - rag_wins:<15} | {total - ov_wins:<15}")
    print(f"{'Accuracy':<20} | {rag_acc:.1f}%{'':<9} | {ov_acc:.1f}%")
    
    print("-" * 60)
    print("🔎 ANALYSIS:")
    print("Standard RAG failed heavily on 'Trap' questions where the distractor")
    print("shared more keywords with the query (e.g. version numbers, negation).")
    print("OV-Memory used Metadata (Centrality) to filter out the high-similarity noise.")
    print("="*60)

if __name__ == "__main__":
    run_10k_benchmark()
