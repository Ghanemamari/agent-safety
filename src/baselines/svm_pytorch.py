"""
SVM via PyTorch (Hinge Loss)
=============================
Replaces scikit-learn's LinearSVC directly in PyTorch bypassing
any missing DLL/environment issues on Windows.

Usage:
    from svm_pytorch import train_svm_pytorch
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from src.eval.metrics import compute_metrics

def train_svm_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 500,
    lr: float = 0.01,
    C: float = 1.0,
    batch_size: int = 256,
    device: str = "cpu"
) -> Dict:
    """
    Train a Linear SVM using Hinge Loss in PyTorch.
    y_train/y_test should be in {0, 1}. Internally mapped to {-1, 1}.
    """
    # Map labels from {0, 1} to {-1, 1} for hinge loss
    y_train_svm = np.where(y_train == 0, -1.0, 1.0)
    y_test_svm = np.where(y_test == 0, -1.0, 1.0)

    # Convert to FloatTensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train_svm, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Linear layer: y = w*x + b
    model = nn.Linear(X_tr.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Dataloader
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb).squeeze()
            
            # Hinge loss: max(0, 1 - y_true * y_pred)
            hinge_loss = torch.mean(torch.clamp(1 - yb * outputs, min=0))
            
            # L2 regularization (equivalent to SVM C parameter inversion)
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.sum(param ** 2)
                
            loss = C * hinge_loss + 0.5 * l2_reg
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        # Get raw decision values (distance to margin)
        # Note: SVM outputs are strictly outside [0, 1], but compute_metrics might expect probs.
        # We can map them via sigmoid just to give a stable ranking for AUROC.
        outputs_test = model(X_te).squeeze()
        
        # Calculate raw accuracy directly from sign
        preds = torch.sign(outputs_test).cpu().numpy()
        # map back {-1, 1} to {0, 1}
        preds = np.where(preds == -1, 0, 1)
        
        # Pseudo-probabilities for AUROC using Sigmoid (Platt scaling approximation)
        probs = torch.sigmoid(outputs_test).cpu().numpy()

    # Calculate metrics
    metrics = compute_metrics(y_test, probs)
    metrics["accuracy"] = float(np.mean(preds == y_test))
    
    return metrics
