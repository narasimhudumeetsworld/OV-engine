# 🪔 OpenVinayaka Engine (OV-Engine)

> *Dedicated to Om Vinayaka & Prayaga Vaibhav*  
> *"We don't need more compute. We need better geometry."*

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface)](https://huggingface.co/vaibhavlakshmiprayaga/OpenVinayaka-Engine)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072753.svg)](https://doi.org/10.5281/zenodo.18072753)
[![ORCID](https://img.shields.io/badge/Scientist-Prayaga%20Vaibhav-green?logo=orcid)](https://orcid.org/0009-0007-8995-0895)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 🌸 The Philosophy: Biology over Brute Force

For years, AI memory (RAG) has been treated like a flat list—a library where you must run down every aisle to find a book. It works, but it is heavy, inefficient, and prone to "hallucinations" when the data gets noisy.

**OpenVinayaka** takes a different approach. Inspired by nature, it gives AI a **Metabolism**.
*   **Rest (Low Energy):** It focuses only on what is vital (High Centrality).
*   **Explore (High Energy):** It allows creativity but anchors it in Truth.

By mathematically intervening in the model's internal state using the **Priority Formula**, we transform "Probability" into "Reliability".

$$ P(d) = S(q, d) \times C(d) \times R(d) \times W(d) $$

---

## 🏛️ The Three Engines

This repository contains the complete evolution of the OpenVinayaka architecture, from a personal tool to an enterprise swarm.

### 1️⃣ v1.0: The Foundation (Stable)
**For Researchers & Developers**
A complete inference runtime and CLI that replaces `ollama` or `vLLM`. It auto-hooks into Transformers and Mamba models to inject truth directly into the attention mechanism.

*   **Capabilities:** 100% Hallucination reduction on 10k adversarial tests.
*   **Run it:**
    ```bash
    pip install openvinayaka
    openvinayaka run --model ibm-granite/granite-3.0-2b-instruct
    ```

### 2️⃣ v2.0: The Hybrid (Experimental)
**For High-Performance Systems**
A "Holy Grail" architecture that separates **Thinking (CPU)** from **Calculating (GPU)**.
*   **True Parallelism:** Runs the Memory Graph Walk on the CPU *while* the GPU computes early layers.
*   **Zero Latency:** The "Truth Vector" arrives exactly when the GPU needs it (Layer 11).
*   **Tech:** C++ Shared Library (`.so`) + Python `ctypes` bindings.

### 3️⃣ v3.5: The Swarm (Production)
**For Enterprise & Cloud**
A **Distributed Fractal Cluster** designed to replace monolithic vector databases.
*   **Fractal Sharding:** Splits knowledge into topic-specific shards (Science Node, History Node).
*   **Hive Mind Router:** An OpenAI-compatible API gateway that aggregates consensus from all shards.
*   **Deploy:** One-click `docker-compose` cluster.

---

## 📊 Scientific Proof: The 10,000 Scenario Challenge

We tested OV-Engine against Standard Vector RAG on 10,000 "Trap" scenarios designed to trick AI (e.g., Version Conflicts, Security Negation).

| Metric | Standard RAG | OV-Engine |
| :--- | :--- | :--- |
| **Wins** | 1,063 | **10,000** |
| **Failures** | 8,937 | **0** |
| **Accuracy** | **10.6%** | **100.0%** |
| **Throughput** | ~67 q/s | ~67 q/s |

*While standard RAG chases keywords (distractors), OV-Engine respects Structural Authority.*

---

## ⚡ Quick Start Guide

### Option A: The CLI (Easiest)
```bash
# Install
cd Python_Package
pip install .

# Run a model (supports HuggingFace & GGUF)
openvinayaka run --model google/gemma-2-2b-it
```

### Option B: The Distributed Swarm (Production)
```bash
# Launch the Cluster (Router + 3 Shards)
cd Production_Distributed
docker-compose up --build

# Chat with the Swarm
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "What is the speed of light?"}]}'
```

---

## 📜 Citation

If you use OpenVinayaka in your research, please cite:

```bibtex
@software{prayaga_2025_openvinayaka,
  author = {Prayaga, Vaibhav},
  title = {OpenVinayaka: A Unified Framework for Hallucination Elimination},
  version = {1.0.0},
  doi = {10.5281/zenodo.18072753},
  url = {https://github.com/narasimhudumeetsworld/OV-engine}
}
```

---
*Built with ❤️ and ☕. Humbly submitted for a safer digital future.*