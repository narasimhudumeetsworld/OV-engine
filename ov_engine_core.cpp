#include "../kernels/ov_universal_kernel.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

// Mock Tensors
using Tensor1D = std::vector<float>;
using Tensor2D = std::vector<std::vector<float>>;

enum ArchType {
    TRANSFORMER_STANDARD,  // Llama, Gemma, GPT
    SSM_MAMBA,            // Mamba 1/2
    HYBRID_JAMBA,         // Transformer + Mamba
    MOE_DEEPSEEK          // Mixture of Experts
};

class OVEngineCore {
public:
    OVEngineCore(ArchType arch) : architecture(arch) {
        std::cout << "⚙️  Initializing OV-Engine Core for Architecture: ";
        switch(arch) {
            case TRANSFORMER_STANDARD: std::cout << "Standard Transformer"; break;
            case SSM_MAMBA: std::cout << "State Space Model (Mamba)"; break;
            case HYBRID_JAMBA: std::cout << "Hybrid (Jamba/Samba)"; break;
            case MOE_DEEPSEEK: std::cout << "Mixture of Experts (MoE)"; break;
        }
        std::cout << std::endl;
        
        // Setup Dummy "Truth" from OV-Memory
        // In prod, this comes from the Graph Walk (ov_graph.cpp)
        active_memory.truth_vector = {1.0f, 1.0f, 1.0f, 1.0f};
        active_memory.confidence = 0.95f; // Very high confidence fact found
        active_memory.is_active = true;
    }

    void run_inference_step() {
        std::cout << "\n▶️  Running Inference Step..." << std::endl;

        switch(architecture) {
            case TRANSFORMER_STANDARD:
                simulate_transformer_step();
                break;
            case SSM_MAMBA:
                simulate_mamba_step();
                break;
            case HYBRID_JAMBA:
                simulate_mamba_step();      // Layer N (SSM)
                simulate_transformer_step(); // Layer N+1 (Attention)
                break;
            case MOE_DEEPSEEK:
                simulate_moe_step();
                break;
        }
    }

private:
    ArchType architecture;
    OV_Kernel::MemoryContext active_memory;

    void print_vec(const std::string& label, const Tensor1D& v) {
        std::cout << "   " << label << ": [ ";
        for (size_t i = 0; i < std::min(v.size(), (size_t)4); ++i) 
            std::cout << std::fixed << std::setprecision(2) << v[i] << " ";
        std::cout << "...]" << std::endl;
    }

    void simulate_transformer_step() {
        // Mock Attention Scores (0-9)
        // Index 0 is our "Truth Context", others are noise
        Tensor1D scores = {0.5f, 1.2f, 0.8f, 0.3f, 1.1f};
        
        std::cout << "   [Attention] Raw Scores (Truth at idx 0 is low):" << std::endl;
        print_vec("Scores", scores);

        // Apply Intervention
        std::cout << "   💉 Applying OV-Attention Bias..." << std::endl;
        OV_Kernel::apply_attention_bias(scores, active_memory, 0, 1, 8.0f);

        print_vec("Result", scores);
        if (scores[0] > 5.0f) std::cout << "   ✅ Attention successfully steered to Truth." << std::endl;
    }

    void simulate_mamba_step() {
        // Mock Hidden State (Drifting to hallucination -0.5)
        Tensor1D state = {-0.5f, -0.6f, -0.4f, -0.5f};
        
        std::cout << "   [SSM State] Current Drifting State:" << std::endl;
        print_vec("State", state);

        // Apply Intervention
        std::cout << "   💉 Applying OV-State Correction..." << std::endl;
        OV_Kernel::apply_state_correction(state, active_memory, 0.4f);

        print_vec("Result", state);
        if (state[0] > 0.0f) std::cout << "   ✅ State vector pulled towards Truth." << std::endl;
    }

    void simulate_moe_step() {
        // Mock Router Logits (Expert 0=Fact, Expert 1=Fiction, Expert 2=Trash)
        Tensor1D logits = {0.5f, 2.0f, 0.5f}; // Router prefers Expert 1 (Fiction)
        std::vector<int> fact_experts = {0}; // Expert 0 is the "Truth" expert

        std::cout << "   [MoE Router] Expert 1 (Fiction) selected:" << std::endl;
        print_vec("Logits", logits);

        // Apply Intervention
        std::cout << "   💉 Applying OV-Router Bias..." << std::endl;
        OV_Kernel::apply_router_bias(logits, fact_experts, active_memory, 5.0f);

        print_vec("Result", logits);
        if (logits[0] > logits[1]) std::cout << "   ✅ Router forced to select Factual Expert." << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== OpenVinayaka Engine v1.0 (Full Version) ===\n" << std::endl;

    // Run tests for all supported architectures
    {
        OVEngineCore engine(TRANSFORMER_STANDARD);
        engine.run_inference_step();
    }
    std::cout << "\n----------------------------------------\n";
    {
        OVEngineCore engine(SSM_MAMBA);
        engine.run_inference_step();
    }
    std::cout << "\n----------------------------------------\n";
    {
        OVEngineCore engine(MOE_DEEPSEEK);
        engine.run_inference_step();
    }
    std::cout << "\n----------------------------------------\n";
    {
        OVEngineCore engine(HYBRID_JAMBA);
        engine.run_inference_step();
    }

    return 0;
}
