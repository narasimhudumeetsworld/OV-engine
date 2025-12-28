# OV-GGUF Specification (v1.0.0-Draft)
**License:** MIT License  
**Author:** Prayaga Vaibhav (Akka) & Open Vinayaka Project

## 1. Abstract
OV-GGUF (Open Vinayaka GGUF) is an extension of the GGUF binary format designed to embed **Hierarchical Memory Graphs** directly into the model file. It enables "Hybrid Inference" where the GPU processes logic while the CPU handles structural knowledge retrieval, eliminating the "Lost in the Middle" phenomenon and reducing hallucinations via structural grounding.

## 2. File Structure Extensions
OV-GGUF adds a new metadata section and a dedicated memory tensor block to the standard GGUF container.

### 2.1 Header Flags
*   `OV_MEMORY_ENABLED` (bool): Signals the inference engine to initialize the OV-Graph walker.
*   `OV_EMBED_DIM` (uint32): Dimension of the memory embeddings (e.g., 128, 768).
*   `OV_LAYERS` (array): List of layer indices (e.g., `[10, 20]`) where memory injection occurs.

### 2.2 Memory Block (`OV_MEM`)
Instead of standard tensors, this block stores the **Graph Topology** in a compressed sparse row (CSR) format:
*   `OV_NODES`: Array of Node IDs and Quantized Embeddings (Int8/Int4).
*   `OV_EDGES`: Adjacency matrix representing the "Centrality" and "Resonance" links.
*   `OV_METADATA`: Packed Centrality (C), Recency (R), and Weight (W) scores.

## 3. The "Hybrid Inference" Loop
Standard inference is: `Input -> GPU(All Layers) -> Output`.
OV-GGUF inference is a **Ping-Pong** process:

1.  **Layers 0-9 (GPU):**
    *   GPU computes basic token embeddings and syntax attention.
    *   *Simultaneously*, CPU "wakes up" relevant nodes in the `OV_MEM` graph based on the input tokens.

2.  **Layer 10 (CPU Injection - JIT):**
    *   The GPU pauses (or computes in parallel).
    *   The CPU calculates the Priority Score: `P = S * C * R * W`.
    *   Top-K "Truth Embeddings" are injected directly into the **KV-Cache** of Layer 10.
    *   **Effect:** The model "remembers" the fact as if it had just seen it, resetting drift.

3.  **Layers 11-End (GPU):**
    *   GPU resumes computation. The attention heads in these layers now attend to the injected KV-Cache entries.
    *   Output is generated with high factual grounding.

## 4. Drift Reduction Mechanism
Standard models drift because errors accumulate in the hidden states (The "Hallucination Snowball").
OV-GGUF prevents this by **overwriting** drifting hidden states with **Ground Truth** from the `OV_MEM` block at the specified `OV_LAYERS`.

## 5. Hardware Benefits
*   **Lower VRAM:** Facts are stored in System RAM (CPU), not VRAM. You can run "smarter" models on smaller GPUs.
*   **Energy Efficient:** Graph traversal (CPU) consumes less power than massive Matrix Multiplications (GPU) for retrieval tasks.

## 6. Implementation Guide (C++)
To support OV-GGUF in engines like `llama.cpp`:
1.  Extend `gguf_context` to parse `OV_MEM` blocks.
2.  Implement `ov_graph_walk()` kernel for CPU.
3.  Add `ov_inject_kv_cache()` function in the `llm_eval` loop.

---
*Om Vinayaka - Dedicated to a safer, more reliable digital future.*
