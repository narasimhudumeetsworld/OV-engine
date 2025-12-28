#ifndef OV_UNIVERSAL_KERNEL_H
#define OV_UNIVERSAL_KERNEL_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// OpenVinayaka Universal Kernel (v1.0)
// Supports: Transformer, Mamba (SSM), MoE, and Hybrid Architectures.
// Goal: Mathematical Hallucination Elimination via Internal State Steering.

namespace OV_Kernel {

    // --- Core Data Structures ---
    struct MemoryContext {
        std::vector<float> truth_vector; // The "Right Answer" embedding
        float confidence;                // P = S * C * R * W (0.0 to 1.0)
        bool is_active;                  // Is there relevant memory to inject?
    };

    // --- 1. Transformer Intervention (Attention Bias) ---
    // Target: QK^T matrix (Pre-Softmax Attention Scores)
    // Effect: "Blinders" that force the model to look at the Truth Context.
    static void apply_attention_bias(
        std::vector<float>& attention_scores, 
        const MemoryContext& mem, 
        int context_start_idx,
        int context_len,
        float strength = 10.0f
    ) {
        if (!mem.is_active || mem.confidence < 0.5f) return;

        // Apply bias only to the tokens representing the injected truth
        for (int i = context_start_idx; i < context_start_idx + context_len; ++i) {
            if (i < attention_scores.size()) {
                // Log-space bias addition (equivalent to multiplying probability)
                attention_scores[i] += (strength * mem.confidence);
            }
        }
    }

    // --- 2. State Space Model (Mamba/Jamba) Intervention ---
    // Target: Hidden State (h_t) in the SSM recurrence
    // Effect: "Drift Correction" - Pulls the state vector back to the Truth.
    static void apply_state_correction(
        std::vector<float>& hidden_state, 
        const MemoryContext& mem,
        float alpha_base = 0.3f
    ) {
        if (!mem.is_active || mem.confidence < 0.5f) return;

        float alpha = alpha_base * mem.confidence; // Correction strength
        
        for (size_t i = 0; i < hidden_state.size(); ++i) {
            // Linear Interpolation (LERP) between Hallucination and Truth
            float current = hidden_state[i];
            float target = mem.truth_vector[i];
            hidden_state[i] = ((1.0f - alpha) * current) + (alpha * target);
        }
    }

    // --- 3. Mixture of Experts (MoE) Intervention ---
    // Target: Router Logits (Gate)
    // Effect: "Expert Steering" - Biases the router to select "Fact/Reasoning" experts 
    // instead of "Creative/Fiction" experts when Truth is present.
    static void apply_router_bias(
        std::vector<float>& router_logits,
        const std::vector<int>& factual_expert_indices,
        const MemoryContext& mem,
        float strength = 5.0f
    ) {
        if (!mem.is_active || mem.confidence < 0.6f) return;

        for (int expert_idx : factual_expert_indices) {
            if (expert_idx < router_logits.size()) {
                router_logits[expert_idx] += (strength * mem.confidence);
            }
        }
    }

    // --- 4. Titans / Neural Memory Intervention ---
    // Target: Persistent Memory Bank (M_t)
    // Effect: "Memory Overwrite" - Directly writes the Truth into the long-term neural memory slot.
    static void apply_memory_overwrite(
        std::vector<std::vector<float>>& memory_bank,
        const MemoryContext& mem,
        int write_head_idx
    ) {
        if (!mem.is_active || mem.confidence < 0.8f) return;

        // Hard overwrite of the memory slot at the write head
        if (write_head_idx < memory_bank.size()) {
            memory_bank[write_head_idx] = mem.truth_vector;
        }
    }
}

#endif // OV_UNIVERSAL_KERNEL_H
