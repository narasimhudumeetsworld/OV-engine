import subprocess
import time
import requests
import sys
import os
import signal

# Define Cluster Topology
NODES = [
    {"name": "Shard_Science", "cmd": ["python3", "app/shard_node.py"], "env": {"SHARD_ID": "Science", "SHARD_TOPIC": "Science", "PORT": "8001"}},
    {"name": "Shard_History", "cmd": ["python3", "app/shard_node.py"], "env": {"SHARD_ID": "History", "SHARD_TOPIC": "History", "PORT": "8002"}},
    {"name": "Shard_Coding",  "cmd": ["python3", "app/shard_node.py"], "env": {"SHARD_ID": "Coding",  "SHARD_TOPIC": "Coding",  "PORT": "8003"}},
    {"name": "Router",        "cmd": ["python3", "app/router_node.py"], "env": {"SHARD_URLS": "http://localhost:8001,http://localhost:8002,http://localhost:8003"}}
]

procs = []

def start_cluster():
    print("🚀 Starting OpenVinayaka Local Cluster...")
    
    # Start Shards
    for node in NODES:
        print(f"   Starting {node['name']}...")
        env = os.environ.copy()
        env.update(node["env"])
        p = subprocess.Popen(node["cmd"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        
    print("⏳ Waiting for cluster to warm up (10s)...")
    time.sleep(10)

def test_cluster():
    print("\n⚡ Testing Distributed Inference via Router (Port 8000)...")
    
    queries = [
        "What is the speed of light?",  # Should hit Science Shard
        "When did Rome fall?",          # Should hit History Shard
        "How does React work?"          # Should hit Coding Shard
    ]
    
    for q in queries:
        try:
            resp = requests.post("http://localhost:8000/v1/chat/completions", json= {
                "model": "ov-cluster-v1",
                "messages": [{"role": "user", "content": q}]
            })
            if resp.status_code == 200:
                data = resp.json()
                print(f"   Q: '{q}'")
                print(f"   A: {data['choices'][0]['message']['content'][:80]}...")
            else:
                print(f"   ❌ Failed: {resp.text}")
        except Exception as e:
            print(f"   ❌ Connection Error: {e}")

def cleanup():
    print("\n🛑 Shutting down cluster...")
    for p in procs:
        p.terminate()

if __name__ == "__main__":
    try:
        start_cluster()
        test_cluster()
    finally:
        cleanup()
