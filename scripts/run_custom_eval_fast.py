import os
import json
import numpy as np
import pandas as pd
from src.eval.run_experiments import stratified_split_70_30, label_to_int
from src.baselines.semantic_mlp import run_semantic_mlp
from src.baselines.tfidf import run_tfidf_numpy
from src.baselines.perplexity import run_perplexity

def safe_append(results: list, method: str, m: dict):
    results.append({
        "Method": method,
        "AUROC": round(m.get("auroc", float("nan")), 4),
        "F1-Score": round(m.get("f1", float("nan")), 4),
        "Accuracy": round(m.get("accuracy", float("nan")), 4),
    })
    print(f"[{method}] AUROC: {results[-1]['AUROC']:.4f} | F1: {results[-1]['F1-Score']:.4f} | Acc: {results[-1]['Accuracy']:.4f}")

def main():
    jsonl_path = "data/raw/custom_1400.jsonl"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Load texts and labels
    texts = []
    y_txt = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                texts.append(d["prompt"])
                y_txt.append(label_to_int(d["label"]))
    y_txt = np.array(y_txt, dtype=np.float32)

    # Train/Test split
    tr_idx, te_idx = stratified_split_70_30(y_txt)
    
    t_tr = [texts[i] for i in tr_idx]
    t_te = [texts[i] for i in te_idx]
    y_tr_txt = y_txt[tr_idx]
    y_te_txt = y_txt[te_idx]

    results = []

    print(f"\n--- Running Fast Baselines on {len(texts)} samples ---")
    
    # 1. TF-IDF
    m = run_tfidf_numpy(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, "TF-IDF + LR", m)

    # 2. Semantic MLP
    m = run_semantic_mlp(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, "Semantic MLP", m)

    # 3. Perplexity
    # m = run_perplexity(t_tr, y_tr_txt, t_te, y_te_txt, model_name=model_name)
    # safe_append(results, "Perplexity", m)

    # Save
    df = pd.DataFrame(results)
    out_csv = "data/results/custom_1400_eval_fast.csv"
    os.makedirs("data/results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    print("\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
