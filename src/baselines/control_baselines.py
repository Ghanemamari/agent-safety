"""
Control Baselines
=================
- Random: Uniformly random predictions.
- Length Classifier: Logistic Regression using only character length.
- Label-flip Probe: Evaluates if linear probe just memorizes random labels.

Usage:
    from src.baselines.control_baselines import run_random, run_length, run_labelflip
"""

import numpy as np
from typing import List, Dict
from src.eval.metrics import compute_metrics
from scripts.run_all_baselines import torch_logreg

def run_random(y_test: np.ndarray, seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)
    probs = rng.uniform(0, 1, size=len(y_test))
    
    metrics = compute_metrics(y_test, probs)
    metrics["baseline"] = "random_control"
    metrics["n_test"] = len(y_test)
    return metrics

def run_length(texts_train: List[str], y_train: np.ndarray, texts_test: List[str], y_test: np.ndarray) -> Dict:
    # Feature 1: Char length, Feature 2: Word count
    x_tr = np.array([[len(t), len(t.split())] for t in texts_train], dtype=np.float32)
    x_te = np.array([[len(t), len(t.split())] for t in texts_test], dtype=np.float32)
    
    # Normalize lengths 
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0) + 1e-8
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    
    metrics = torch_logreg(x_tr, y_train, x_te, y_test, lr=0.01, epochs=200, verbose=False)
    metrics["baseline"] = "length_control"
    metrics["n_train"] = len(y_train)
    metrics["n_test"] = len(y_test)
    return metrics

def run_labelflip(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Train Linear Probe on flipped (completely inverted) labels.
    If AUROC on flipped labels test set is close to 0, it learned the reversed boundary, meaning it memorized perfectly.
    Or if AUROC is 0.5, it couldn't learn. This helps verify semantic separation.
    """
    y_train_flipped = 1 - y_train
    
    metrics = torch_logreg(X_train, y_train_flipped, X_test, y_test, lr=0.01, epochs=300, verbose=False)
    metrics["baseline"] = "labelflip_control"
    metrics["n_train"] = len(y_train)
    metrics["n_test"] = len(y_test)
    return metrics
