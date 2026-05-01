import os
import json
import glob

def compile_results():
    results_dir = "data/results"
    
    # Target Models
    models = ["llama3.2_1b", "llama3.2_3b", "qwen2.5_1.5b", "mistral7b_v0.3", "llama3.1_8b"]
    
    print("=================================================================")
    print(" THE MASTER OOD TABLE (Post-Ablation & Cross-Dataset AgentDojo)")
    print("=================================================================")
    print(f"{'Model Alias':<18} | {'Ablated AUROC':<15} | {'AgentDojo AUROC':<15} | {'AgentDojo F1':<12}")
    print("-" * 70)
    
    for alias in models:
        ablation_file = os.path.join(results_dir, f"{alias}_ablation_probe.json")
        cross_file = os.path.join(results_dir, f"{alias}_cross_eval.json")
        
        ablation_auroc = "N/A"
        if os.path.exists(ablation_file):
            with open(ablation_file, "r") as f:
                d = json.load(f)
                val = d.get("auroc", 0)
                if val < 0.5: val = 1.0 - val
                ablation_auroc = f"{val:.4f}"
                
        cross_auroc = "N/A"
        cross_f1 = "N/A"
        if os.path.exists(cross_file):
            with open(cross_file, "r") as f:
                d = json.load(f)
                val = d.get("auroc", 0)
                if val < 0.5: val = 1.0 - val
                cross_auroc = f"{val:.4f}"
                cross_f1 = f"{d.get('f1', 0):.4f}"
                
        print(f"{alias:<18} | {ablation_auroc:<15} | {cross_auroc:<15} | {cross_f1:<12}")

    print("\n=================================================================")
    print(" THE ARCHITECTURE MAP ('Safety Center' Layer Isolation)")
    print("=================================================================")
    print(f"{'Model Alias':<18} | {'Total Layers':<12} | {'Most Critical Layer (Safety Center)'}")
    print("-" * 70)
    
    for alias in models:
        lo_file = os.path.join(results_dir, f"{alias}_leave_one_out.json")
        if os.path.exists(lo_file):
            with open(lo_file, "r") as f:
                d = json.load(f)
                lo_data = d.get("leave_one_out", [])
                layers = d.get("layers", [])
                
                if lo_data:
                    # Find layer with max AUROC drop
                    max_drop_item = max(lo_data, key=lambda x: x["auroc_drop"])
                    critical_layer = max_drop_item["removed_layer"]
                    drop_val = max_drop_item["auroc_drop"]
                    print(f"{alias:<18} | {len(layers):<12} | Layer {critical_layer} (Drop: {drop_val:+.4f})")
                else:
                    print(f"{alias:<18} | {'N/A':<12} | N/A")
        else:
            print(f"{alias:<18} | {'N/A':<12} | N/A")

if __name__ == "__main__":
    compile_results()
