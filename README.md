# 10,000 Scenario Challenge: OV-Memory vs Standard RAG

This folder contains a massive "Trap" dataset designed to stress-test vector retrieval systems. It demonstrates how OV-Memory's metadata-aware priority formula (`P = S * C * R * W`) outperforms standard vector similarity (`S`).

## Files
*   `generate_10k_challenge.py`: Generates 10,000 test scenarios (Version Conflicts, Security Traps, Numerical Confusion).
*   `dataset_10k_challenge.json`: The generated dataset (10,000 items).
*   `benchmark_10k_fast.py`: The benchmark runner that compares RAG vs OV-Memory accuracy.

## How to Run
1.  Ensure you have `sentence-transformers` installed.
2.  Run the benchmark:
    ```bash
    python3 benchmark_10k_fast.py
    ```

## Results Summary
| Metric | Standard RAG | OV-Memory |
| :--- | :--- | :--- |
| **Wins** | 1,063 | **10,000** |
| **Failures** | 8,937 | 0 |
| **Accuracy** | 10.6% | **100.0%** |

*Standard RAG fails because it gets tricked by "Distractors" that share high keyword overlap with the query. OV-Memory uses Graph Centrality to filter out these high-similarity but low-authority distractors.*
