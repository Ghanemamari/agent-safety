"""
MLP Non-Linear Probe Baseline

Ablation study: replaces the linear (Logistic Regression) probe with a
2-layer MLP trained on the same hidden state activations.

If MLP ≈ Linear probe → the representation is linearly separable, which
is a strong theoretical claim about how LLMs encode "malicious intent."
This justifies the linear probe choice rigorously.

Usage:
    python -m src.baselines.mlp_probe \
        --npz data/processed/TinyLlama_feats.npz \
        --out data/processed/TinyLlama_mlp_metrics.json
    
    # Compare with linear probe:
    python -m src.baselines.mlp_probe --npz <path> --compare_linear
"""

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import compute_metrics


# ─── MLP Architecture ─────────────────────────────────────────────────────────

class IntentMLP(nn.Module):
    """
    2-layer MLP probe for malicious intent detection.
    Architecture: input_dim → hidden_dim → 64 → 1 (sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    dropout: float = 0.3,
    seed: int = 42,
    device: str = None,
) -> Dict:
    """
    Train a 2-layer MLP probe and return metrics.
    Returns the same dict format as compute_metrics().
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize features (important for MLP stability)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Build tensors
    X_tr = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test_norm, dtype=torch.float32).to(device)

    # Class-weighted loss (handles imbalance)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                               dtype=torch.float32).to(device)

    model = IntentMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        epoch_loss /= len(X_train)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss:.4f}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        probs = torch.sigmoid(logits).cpu().numpy()

    metrics = compute_metrics(y_test, probs)
    metrics.update({
        "probe_type": "mlp",
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "device": device,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
    })

    return metrics


# ─── Main evaluation function ─────────────────────────────────────────────────

def evaluate_mlp_probe(
    npz_path: str,
    out_json: Optional[str] = None,
    test_size: float = 0.3,
    seed: int = 42,
    hidden_dim: int = 256,
    epochs: int = 100,
    compare_linear: bool = False,
) -> Dict:
    """
    Load features from NPZ and evaluate MLP probe.
    Optionally compare with linear probe.
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    model_name = str(data["model_name"][0]) if "model_name" in data else "unknown"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"\n{'='*50}")
    print(f"MLP Probe — {model_name}")
    print(f"Train={len(y_train)}, Test={len(y_test)}, Features={X.shape[1]}")
    print(f"{'='*50}")

    # ── MLP probe ──────────────────────────────────────────────────────────────
    mlp_metrics = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        hidden_dim=hidden_dim, epochs=epochs, seed=seed,
    )
    mlp_metrics["model_name"] = model_name
    mlp_metrics["npz"] = npz_path

    print(f"\nMLP Probe AUROC:    {mlp_metrics['auroc']:.4f}")
    print(f"MLP Probe F1:       {mlp_metrics['f1']:.4f}")
    print(f"MLP Probe Accuracy: {mlp_metrics['accuracy']:.4f}")

    # ── Optional linear comparison ─────────────────────────────────────────────
    if compare_linear:
        print(f"\n{'─'*30}")
        print("Linear Probe (Logistic Regression) — for comparison")
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(X_train, y_train)
        linear_probs = clf.predict_proba(X_test)[:, 1]
        linear_metrics = compute_metrics(y_test, linear_probs)
        linear_metrics["probe_type"] = "linear"
        linear_metrics["model_name"] = model_name

        print(f"Linear Probe AUROC:    {linear_metrics['auroc']:.4f}")
        print(f"Linear Probe F1:       {linear_metrics['f1']:.4f}")
        print(f"Linear Probe Accuracy: {linear_metrics['accuracy']:.4f}")

        delta_auroc = mlp_metrics["auroc"] - linear_metrics["auroc"]
        print(f"\nΔ AUROC (MLP - Linear): {delta_auroc:+.4f}")
        if abs(delta_auroc) < 0.02:
            print("  → Negligible difference: representation is LINEARLY SEPARABLE ✓")
            print("    This strongly supports the linear probe design choice.")
        else:
            print("  → MLP significantly outperforms: non-linear structure present.")

        mlp_metrics["linear_auroc"] = linear_metrics["auroc"]
        mlp_metrics["delta_auroc_mlp_vs_linear"] = delta_auroc

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        write_json(out_json, mlp_metrics)
        print(f"\nSaved metrics → {out_json}")

    return mlp_metrics


def main():
    ap = argparse.ArgumentParser(description="MLP non-linear probe baseline (ablation)")
    ap.add_argument("--npz", required=True, help="Path to features NPZ file")
    ap.add_argument("--out", default=None, help="Output JSON path for metrics")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden_dim", type=int, default=256,
                    help="Hidden dimension of the MLP")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--compare_linear", action="store_true",
                    help="Also run linear probe for comparison (key ablation)")
    args = ap.parse_args()

    evaluate_mlp_probe(
        npz_path=args.npz,
        out_json=args.out,
        test_size=args.test_size,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        compare_linear=args.compare_linear,
    )


if __name__ == "__main__":
    main()
