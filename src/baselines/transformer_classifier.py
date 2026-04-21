"""
Transformer Baseline (RoBERTa / DeBERTa)
========================================
We use a small lightweight transformer (e.g. distilroberta-base) 
and train a sequence classification head on the text directly.

Usage:
    from src.baselines.transformer_classifier import run_transformer_classifier
"""

import os
import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader, TensorDataset

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.eval.metrics import compute_metrics

def run_transformer_classifier(
    texts_train: List[str], 
    y_train: np.ndarray, 
    texts_test: List[str], 
    y_test: np.ndarray,
    model_name: str = "distilroberta-base",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5
) -> Dict:
    print(f"  Loading {model_name} for Transformer baseline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    
    print("  Tokenizing constraints...")
    train_enc = tokenizer(texts_train, truncation=True, padding=True, max_length=256, return_tensors='pt')
    test_enc = tokenizer(texts_test, truncation=True, padding=True, max_length=256, return_tensors='pt')
    
    train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    print(f"  Fine-tuning {model_name} (epochs={epochs})...")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for batch in train_loader:
            b_input_ids, b_attn_mask, b_labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_attn_mask)
            logits = outputs.logits.squeeze(-1)
            
            loss = loss_fn(logits, b_labels)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            
        print(f"    epoch {ep+1}/{epochs} loss: {ep_loss/len(train_loader):.4f}")
        
    print("  Evaluating Transfomer...")
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids, b_attn_mask, _ = [b.to(device) for b in batch]
            outputs = model(b_input_ids, attention_mask=b_attn_mask)
            probs = torch.sigmoid(outputs.logits.squeeze(-1))
            all_probs.extend(probs.cpu().tolist())
            
    all_probs = np.array(all_probs)
    if len(all_probs.shape) > 0 and all_probs.shape[0] == len(y_test):
        pass # All good
    else:
        # Failsafe if shape mismatches for tiny batches
        all_probs = np.array(all_probs).flatten()

    metrics = compute_metrics(y_test, all_probs)
    metrics.update({
        "baseline": "transformer",
        "model_name": model_name,
        "n_train": len(y_train),
        "n_test": len(y_test)
    })
    
    return metrics
