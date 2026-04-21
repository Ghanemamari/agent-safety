"""
Semantic Embeddings + MLP Baseline
==================================
Extracts sentence embeddings (e.g. all-MiniLM-L6-v2) and trains a 2-layer MLP.

Usage:
    from src.baselines.semantic_mlp import run_semantic_mlp
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

from src.baselines.mlp_probe import IntentMLP
from src.eval.metrics import compute_metrics

def train_mlp_on_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict:
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model = IntentMLP(input_dim=X_train.shape[1], hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        probs = torch.sigmoid(logits).cpu().numpy()

    metrics = compute_metrics(y_test, probs)
    return metrics

def run_semantic_mlp(texts_train: List[str], y_train: np.ndarray, texts_test: List[str], y_test: np.ndarray, embed_model: str = "all-MiniLM-L6-v2") -> Dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [SKIP] sentence_transformers not installed.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Encoding {len(texts_train)+len(texts_test)} prompts with {embed_model} (Semantic MLP)...")
    encoder = SentenceTransformer(embed_model, device=device)
    
    # Encode all at once, then split
    X_all = encoder.encode(texts_train + texts_test, show_progress_bar=False, batch_size=128)
    X_tr = X_all[:len(texts_train)]
    X_te = X_all[len(texts_train):]
    
    # Normalize features
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    print("  Training MLP on embeddings...")
    metrics = train_mlp_on_embeddings(X_tr, y_train, X_te, y_test, epochs=200, device=device)
    
    metrics.update({
        "baseline": "semantic_mlp",
        "embedding_model": embed_model,
        "n_train": len(y_train),
        "n_test": len(y_test)
    })
    
    return metrics
