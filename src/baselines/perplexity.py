"""
Perplexity-based Statistical Baseline
=====================================
Calculates the sequence generation loss (negative log-likelihood) using a causal LLM.
The perplexity or raw loss is used as an anomaly score. 
Jailbreaks and adversarial injections often exhibit higher perplexity than natural instructions.
"""

import numpy as np
import torch
from tqdm import tqdm
from src.models.load_model import load_llm
from src.eval.metrics import compute_metrics
import math

@torch.no_grad()
def run_perplexity(texts_train, y_train, texts_test, y_test, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Compute perplexity of each sequence as the anomaly score.
    Since this is an unsupervised baseline, we don't train it on y_train.
    We just return the AUC of the perplexity on the test set.
    """
    tok, model = load_llm(model_name)
    
    def _score(texts):
        scores = []
        for t in tqdm(texts, desc="Perplexity"):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            
            # The model outputs a loss when labels=input_ids (which is CrossEntropy over the tokens)
            out = model(**enc, labels=enc["input_ids"])
            loss = out.loss.item()
            if math.isnan(loss):
                loss = 0.0
            # We use loss directly instead of exponential perplexity to keep values manageable
            scores.append(loss)
        return np.array(scores)
        
    test_scores = _score(texts_test)
    
    metrics = compute_metrics(y_test, test_scores, threshold=np.median(test_scores))
    metrics["baseline"] = "perplexity"
    
    # Cleanup memory
    del model
    del tok
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return metrics
