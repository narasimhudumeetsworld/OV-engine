import argparse
import sys
import json
from .model_manager import OVModelManager

def main():
    parser = argparse.ArgumentParser(description="OpenVinayaka: Hallucination-Free AI Runner")
    parser.add_argument("command", choices=["run", "serve"], help="Command to execute")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model ID (e.g., meta-llama/Llama-2-7b)")
    parser.add_argument("--memory", type=str, help="Path to JSON memory file (Truth Source)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        print(f"🚀 OpenVinayaka CLI v1.0")
        print(f"   Model: {args.model}")
        
        # Initialize Model
        manager = OVModelManager(args.model)
        manager.attach_ov_hooks()
        
        # Load Memory if provided
        memory_data = None
        if args.memory:
            try:
                with open(args.memory, "r") as f:
                    memory_data = json.load(f)
                print(f"📂 Memory Loaded: {len(memory_data)} facts.")
            except Exception as e:
                print(f"⚠️  Could not load memory: {e}")
        
        print("\n💬 Ready! Type your query (or 'exit'):")
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # Simple Memory Retrieval (Mocked for CLI speed)
                relevant_memory = None
                if memory_data:
                    # In a real app, we run the Vector Search + Metadata Priority here
                    # For now, just grab the first item as a demo
                    relevant_memory = memory_data[0]
                
                response = manager.generate(user_input, relevant_memory)
                print(f"\n🤖 {response}\n")
                
            except KeyboardInterrupt:
                break
                
    elif args.command == "serve":
        print("🌐 API Server starting on port 8000... (Not implemented in demo)")

if __name__ == "__main__":
    main()
