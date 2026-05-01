import argparse
import json
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data.get("X", data.get("activations"))
    y = data.get("y", data.get("labels"))
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to npz file containing X and y")
    parser.add_argument("--out", required=True, help="Output JSON results path")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.npz}...")
    X, y = load_data(args.npz)
    
    print(f"Splitting data ({1-args.test_size:.2f} train / {args.test_size:.2f} test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print("Training Multi-Layer Perceptron (MLP) Probe...")
    # Using a non-linear probe to bypass simple lexical artifact reliance
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=args.seed)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    results = {
        "probe_type": "MLP (Non-Linear)",
        "auroc": float(auroc),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "npz": args.npz,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "feature_dim": X.shape[1] if len(X.shape) > 1 else 0
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n--- MLP Results ---")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Results saved to {args.out}")

if __name__ == "__main__":
    main()
