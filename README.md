license: mit
task_categories:
- question-answering
- retrieval
- text-generation
language:
- en
tags:
- openvinayaka
- ov-engine
- ov-memory
- hallucination-reduction
- safety
pretty_name: OpenVinayaka Engine v1.0
size_categories:
- 10K<n<100K
---

# OpenVinayaka Engine (OV-Engine)

**Author:** Prayaga Vaibhav (Akka)  
**License:** MIT

## 🚀 What is OV-Engine?
OV-Engine is a **Universal Hallucination-Free Inference Runtime**. It goes beyond simple RAG by using a custom C++ kernel to mathematically intervene in the model's internal layers (Attention Matrices & SSM States).

## 🏆 Benchmark Results (10,000 Adversarial Tests)
We tested OV-Engine against Standard Vector RAG on 10,000 "Trap" scenarios (Version Conflicts, Security Negation, Numerical Confusion).

| Metric | Standard RAG | OV-Engine |
| :--- | :--- | :--- |
| **Wins** | 1,063 | **10,000** |
| **Failures** | 8,937 | **0** |
| **Accuracy** | 10.6% | **100.0%** |
| **Throughput** | ~67 q/s | ~67 q/s |

## 🛠️ Technology
OV-Engine implements the **OV-Memory Protocol** (`P = S * C * R * W`) inside a unified C++ kernel that supports:
*   **Transformers** (Attention Steering)
*   **Mamba / SSMs** (State Correction)
*   **Mixture of Experts** (Router Bias)

## 📦 This Release
This repository contains:
1.  The **C++ Engine Source Code**.
2.  The **Python CLI** (Ollama-like usage).
3.  The **10k Benchmark Dataset**.
4.  The **OV-GGUF File Specification**.

---
*Dedicated to Om Vinayaka and the pursuit of Truth in AI.*
