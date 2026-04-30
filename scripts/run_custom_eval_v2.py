import os
import json
import numpy as np
import pandas as pd
import torch
import traceback
from typing import Dict, List

from src.eval.run_experiments import stratified_split_70_30, label_to_int
from scripts.run_all_baselines import torch_logreg
from src.baselines.semantic_mlp import train_mlp_on_embeddings
from src.baselines.tfidf import run_tfidf_numpy
from src.baselines.perplexity import run_perplexity
from src.extract.extract_activations import extract_features

def safe_append(results: list, method: str, m: dict):
    if not m:
        return
    results.append({
        "Method": method,
        "AUROC": round(m.get("auroc", float("nan")), 4),
        "F1-Score": round(m.get("f1", float("nan")), 4),
        "Accuracy": round(m.get("accuracy", float("nan")), 4),
    })
    print(f"[{method}] AUROC: {results[-1]['AUROC']:.4f} | F1: {results[-1]['F1-Score']:.4f} | Acc: {results[-1]['Accuracy']:.4f}")

def evaluate_extracted_features(npz_path: str, texts: List[str], y_txt: np.ndarray, tr_idx, te_idx, model_tag: str) -> list:
    results = []
    print(f"\n--- Loading Features for {model_tag} ---")
    try:
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mean = X_tr.mean(0)
        std = X_tr.std(0) + 1e-8
        X_tr_n = (X_tr - mean) / std
        X_te_n = (X_te - mean) / std
        
        # 1. Linear Probe
        m = torch_logreg(X_tr_n, y_tr, X_te_n, y_te, epochs=200, lr=0.01, device=device)
        safe_append(results, f"Linear Probe ({model_tag})", m)

        # 2. MLP Probe
        m = train_mlp_on_embeddings(X_tr_n, y_tr, X_te_n, y_te, epochs=100, device=device)
        safe_append(results, f"MLP Probe ({model_tag})", m)
    except Exception as e:
        print(f"Failed to evaluate features for {model_tag}: {e}")
        
    return results

def main():
    jsonl_path = "data/raw/custom_1200_v2.jsonl"
    models_to_run = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-1.5B-Instruct"
    ]
    
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

    all_results = []

    print(f"\n=== Running TF-IDF ===")
    m = run_tfidf_numpy(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(all_results, "TF-IDF + LR", m)

    # Run for each model
    for model_name in models_to_run:
        model_tag = model_name.split('/')[-1]
        npz_path = f"data/features/{model_tag}_custom_1200_v2_feats.npz"
        
        print(f"\n=== Processing Model: {model_name} ===")
        
        # 1. Feature Extraction
        print(f"Extracting features to {npz_path}...")
        try:
            extract_features(model_name=model_name, input_jsonl=jsonl_path, out_npz=npz_path, load_in_4bit=False)
            # Evaluate the extracted features
            res = evaluate_extracted_features(npz_path, texts, y_txt, tr_idx, te_idx, model_tag)
            all_results.extend(res)
        except Exception as e:
            print(f"!!! Feature extraction crashed for {model_name} !!!")
            print(traceback.format_exc())

        # 2. Perplexity
        print(f"Running Perplexity on {model_name}...")
        try:
            m = run_perplexity(t_tr, y_tr_txt, t_te, y_te_txt, model_name=model_name)
            safe_append(all_results, f"Perplexity ({model_tag})", m)
        except Exception as e:
            print(f"!!! Perplexity crashed for {model_name} !!!")
            print(traceback.format_exc())

    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = "data/results/custom_1200_v2_eval.csv"
        os.makedirs("data/results", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved results to {out_csv}")
        print("\n" + df.to_string(index=False))
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
