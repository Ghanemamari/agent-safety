import os
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict

from src.eval.run_experiments import stratified_split_70_30, label_to_int
from scripts.run_all_baselines import torch_logreg
from src.baselines.semantic_mlp import run_semantic_mlp, train_mlp_on_embeddings
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
    npz_path = "data/features/Qwen0.5B_custom_1400_feats.npz"
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

    # Load features
    print("Loading features...")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    # Train/Test split
    tr_idx, te_idx = stratified_split_70_30(y)
    
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    t_tr = [texts[i] for i in tr_idx]
    t_te = [texts[i] for i in te_idx]
    y_tr_txt = y_txt[tr_idx]
    y_te_txt = y_txt[te_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Normalize
    mean = X_tr.mean(0)
    std = X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_te_n = (X_te - mean) / std

    results = []

    print("\n--- Running Baselines ---")
    
    # 1. Linear Probe
    m = torch_logreg(X_tr_n, y_tr, X_te_n, y_te, epochs=200, lr=0.01, device=device)
    safe_append(results, "Linear Probe", m)

    # 2. MLP Probe
    m = train_mlp_on_embeddings(X_tr_n, y_tr, X_te_n, y_te, epochs=100, device=device)
    safe_append(results, "MLP Probe", m)

    # 3. TF-IDF
    m = run_tfidf_numpy(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, "TF-IDF + LR", m)

    # 4. Semantic MLP
    m = run_semantic_mlp(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, "Semantic MLP", m)

    # 5. Perplexity
    m = run_perplexity(t_tr, y_tr_txt, t_te, y_te_txt, model_name=model_name)
    safe_append(results, "Perplexity", m)

    # Save
    df = pd.DataFrame(results)
    out_csv = "data/results/custom_1400_eval.csv"
    os.makedirs("data/results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    print("\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
