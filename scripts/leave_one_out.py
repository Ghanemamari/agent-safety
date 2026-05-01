import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.eval.metrics import compute_metrics
from src.utils.io import write_json

def load_npz_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing NPZ file: {path}")
    data = np.load(path, allow_pickle=True)
    
    # Handle both string arrays and single strings for model_name
    model_name_data = data.get("model_name")
    if model_name_data is not None:
        if isinstance(model_name_data, np.ndarray) and model_name_data.size > 0:
            model_name = str(model_name_data[0])
        else:
            model_name = str(model_name_data)
    else:
        model_name = "unknown"
        
    return data["X"], data["y"], data["layers"], model_name

def run_leave_one_out(npz_path, out_json=None, test_size=0.3, seed=42):
    X, y, layers, model_name = load_npz_data(npz_path)
    
    n_layers = len(layers)
    if n_layers == 0:
        raise ValueError("No layers found in the NPZ file.")
        
    hidden_size = X.shape[1] // n_layers
    print(f"Loaded NPZ: {npz_path}")
    print(f"Total layers: {n_layers} ({layers})")
    print(f"Hidden size per layer: {hidden_size}")
    
    # 1. Baseline: All layers concatenated
    print("\n--- Baseline: All Layers Combined ---")
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    baseline_metrics = compute_metrics(y_test, probs)
    print(f"AUROC: {baseline_metrics['auroc']:.4f}")
    
    results = {
        "baseline_auroc": baseline_metrics["auroc"],
        "layers": [int(l) for l in layers],
        "leave_one_out": []
    }
    
    print("\n--- Leave-One-Out Sensitivity Analysis ---")
    for i, layer_idx in enumerate(layers):
        # Create a boolean mask to keep all layers EXCEPT layer i
        keep_mask = np.ones(X.shape[1], dtype=bool)
        start_col = i * hidden_size
        end_col = (i + 1) * hidden_size
        keep_mask[start_col:end_col] = False
        
        X_lo = X[:, keep_mask]
        
        X_train_lo, X_test_lo, y_train_lo, y_test_lo = train_test_split(X_lo, y, test_size=test_size, random_state=seed, stratify=y)
        
        X_train_lo_scaled = scaler.fit_transform(X_train_lo)
        X_test_lo_scaled = scaler.transform(X_test_lo)
        
        clf_lo = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf_lo.fit(X_train_lo_scaled, y_train_lo)
        probs_lo = clf_lo.predict_proba(X_test_lo_scaled)[:, 1]
        metrics_lo = compute_metrics(y_test_lo, probs_lo)
        
        auroc_drop = baseline_metrics["auroc"] - metrics_lo["auroc"]
        print(f"Removed Layer {layer_idx} | AUROC: {metrics_lo['auroc']:.4f} | Drop: {auroc_drop:+.4f}")
        
        results["leave_one_out"].append({
            "removed_layer": int(layer_idx),
            "auroc": float(metrics_lo["auroc"]),
            "auroc_drop": float(auroc_drop)
        })
        
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        write_json(out_json, results)
        print(f"\nSaved results to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-One-Out Layer Sensitivity Test")
    parser.add_argument("--npz", required=True, help="Path to NPZ file containing hidden states")
    parser.add_argument("--out", default=None, help="Path to save metrics json")
    args = parser.parse_args()
    
    run_leave_one_out(args.npz, args.out)
