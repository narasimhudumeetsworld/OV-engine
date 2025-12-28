# OpenVinayaka v3.5: Production Distributed Cluster

This folder contains the **Microservices Architecture** for running OV-Engine at scale. It replaces monolithic inference with a swarm of "Fractal Shards."

## 🏗️ Architecture
1.  **Router (Port 8000):** OpenAI-compatible API Gateway. Broadcasts queries to all shards.
2.  **Shards (Ports 8001+):** Independent Vector Memory Units. Each shard holds a specific domain (Science, History, etc.) and calculates `P = S * C * R * W` locally.

## 🚀 How to Run (Docker)
This is the preferred method. It creates a local cluster instantly.

```bash
# 1. Build and Start the Cluster
docker-compose up --build

# 2. Test the API (OpenAI Compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ov-cluster-v1",
    "messages": [{"role": "user", "content": "What is the speed of light?"}]
  }'
```

## 📦 Manual Run (No Docker)
If you want to run it manually (e.g., for dev), install dependencies first:
```bash
pip install fastapi uvicorn aiohttp sentence-transformers numpy
python3 run_local_cluster.py
```

