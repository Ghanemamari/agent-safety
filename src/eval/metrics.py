"""
metrics.py – pure NumPy implementation (no sklearn dependency)
"""
from typing import Dict
import numpy as np


def roc_auc_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true   = np.asarray(y_true,   dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    desc_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[desc_idx]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(_trapz(tpr, fpr))


def compute_metrics(
    y_true,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    probs  = np.asarray(probs,  dtype=float)
    preds  = (probs >= threshold).astype(float)

    auroc = roc_auc_numpy(y_true, probs)
    acc   = float((preds == y_true).mean())

    tp = float(((preds == 1) & (y_true == 1)).sum())
    fp = float(((preds == 1) & (y_true == 0)).sum())
    fn = float(((preds == 0) & (y_true == 1)).sum())
    prec   = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)

    return {
        "auroc":     round(auroc,  4),
        "accuracy":  round(acc,    4),
        "f1":        round(f1,     4),
        "precision": round(prec,   4),
        "recall":    round(recall, 4),
        "threshold": threshold,
    }
