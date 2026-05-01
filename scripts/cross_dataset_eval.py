import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.eval.metrics import compute_metrics
from src.utils.io import write_json

def load_npz_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing NPZ file: {path}")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]

def cross_dataset_eval(train_npzs, test_npz, out_json=None):
    print(f"Loading training data from: {train_npzs}")
    X_train_list, y_train_list = [], []
    for p in train_npzs:
        X, y = load_npz_data(p)
        X_train_list.append(X)
        y_train_list.append(y)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    print(f"Loading test data from: {test_npz}")
    X_test, y_test = load_npz_data(test_npz)
    
    print(f"Training set: X={X_train.shape}, y={y_train.shape} (Positives: {y_train.sum()})")
    print(f"Test set: X={X_test.shape}, y={y_test.shape} (Positives: {y_test.sum()})")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_metrics(y_test, probs)
    
    metrics.update({
        "train_datasets": train_npzs,
        "test_dataset": test_npz,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
    })
    
    print("\n--- Cross-Dataset Generalization Results ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        write_json(out_json, metrics)
        print(f"Saved metrics to {out_json}")
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Dataset Generalization Test")
    parser.add_argument("--train", nargs="+", required=True, help="Paths to training NPZ files (e.g. injecagent.npz advbench.npz)")
    parser.add_argument("--test", required=True, help="Path to testing NPZ file (e.g. agentdojo.npz)")
    parser.add_argument("--out", default=None, help="Path to save metrics json")
    args = parser.parse_args()
    
    cross_dataset_eval(args.train, args.test, args.out)
