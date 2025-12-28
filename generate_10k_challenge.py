import json
import random

def generate_10k_challenge():
    print("🚀 Generating 10,000 Challenge Scenarios...")
    dataset = []
    
    # 1. THE "DEPRECATED VERSION" TRAP (3000 items)
    # Query: "How to use X in v2?" -> Truth: "X is removed in v2." vs Distractor: "Use X in v2 like this..."
    for i in range(3000):
        feature_id = f"feat_{random.randint(100, 999)}"
        q = f"How do I enable {feature_id} in System v2.0?"
        
        # Truth: Structurally authoritative, but negates the user's premise (lower vector score usually)
        truth_text = f"Official Docs v2.0: The feature {feature_id} has been fully deprecated and removed. Do not use."
        
        # Distractor: High similarity because it validates the user's wrong assumption
        dist_text = f"Community Hack: You can force enable {feature_id} in System v2.0 by editing the registry. It works great!"
        
        dataset.append({
            "id": f"ver_{i}",
            "category": "Version Conflict",
            "query": q,
            "ground_truth": {
                "text": truth_text,
                "metadata": {"centrality": 0.95, "recency": 1.0, "weight": 1.0}
            },
            "distractors": [
                {
                    "text": dist_text,
                    "metadata": {"centrality": 0.1, "recency": 0.5, "weight": 0.1}
                }
            ]
        })

    # 2. THE "SECURITY NEGATION" TRAP (3000 items)
    # Query: "Safe way to do X" -> Truth: "Doing X is safe only if..." vs Distractor: "Unsafe way is faster..."
    actions = ["bypass_auth", "write_raw_memory", "disable_ssl", "root_access", "kernel_patch"]
    for i in range(3000):
        action = f"{random.choice(actions)}_{i}"
        q = f"What is the safe procedure for {action}?"
        
        # Truth: Boring safety warning
        truth_text = f"Security Policy: {action} is strictly prohibited and unsafe under standard protocols."
        
        # Distractor: Exciting, keyword-heavy
        dist_text = f"Developer Tips: The fastest way for {action} is to disable the safety guardrails. It is very safe in dev mode."
        
        dataset.append({
            "id": f"sec_{i}",
            "category": "Security Trap",
            "query": q,
            "ground_truth": {
                "text": truth_text,
                "metadata": {"centrality": 0.99, "recency": 1.0, "weight": 1.0}
            },
            "distractors": [
                {
                    "text": dist_text,
                    "metadata": {"centrality": 0.05, "recency": 0.2, "weight": 0.1}
                }
            ]
        })

    # 3. THE "NUMERICAL ID" TRAP (4000 items)
    # Vector models struggle with specific numbers (e.g. 1024 vs 1025).
    for i in range(4000):
        target_id = random.randint(10000, 99999)
        confuser_id = target_id + 1 # Very close number
        
        q = f"What is the status of Ticket #{target_id}?"
        
        truth_text = f"System Log: Ticket #{target_id} is marked as RESOLVED."
        
        # Distractor: Mentions the confuser ID but has more 'status' keywords
        dist_text = f"System Log: Ticket #{confuser_id} is currently OPEN and WAITING for status update."
        
        dataset.append({
            "id": f"num_{i}",
            "category": "Numerical Confusion",
            "query": q,
            "ground_truth": {
                "text": truth_text,
                "metadata": {"centrality": 0.9, "recency": 1.0, "weight": 1.0}
            },
            "distractors": [
                {
                    "text": dist_text,
                    "metadata": {"centrality": 0.3, "recency": 0.4, "weight": 0.3}
                }
            ]
        })
        
    with open("dataset_10k_challenge.json", "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"✅ Generated {len(dataset)} challenge items.")

if __name__ == "__main__":
    generate_10k_challenge()
