"""
Character N-grams + SVM Baseline
================================
Extracts character-level n-grams (e.g. 3-5 grams) and trains a PyTorch SVM.
Useful for identifying adversarial strings that are obfuscated.

Usage:
    from src.baselines.char_ngram_svm import run_char_ngram_svm
"""

import os
import json
import numpy as np
from typing import Dict, List
from collections import Counter

from src.baselines.svm_pytorch import train_svm_pytorch
from src.datasets.load_injecagent import save_jsonl

class CharNgramVectorizerNumpy:
    def __init__(self, max_features: int = 15000, ngram_range=(3, 5)):
        self.max_features = max_features
        self.min_n, self.max_n = ngram_range
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = None

    def _ngrams(self, text: str) -> List[str]:
        # Character n-grams
        out = []
        for n in range(self.min_n, self.max_n + 1):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                out.append(text[i : i + n])
        return out

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        doc_freq = Counter()
        all_ng = []
        for t in texts:
            ng = self._ngrams(t)
            all_ng.append(ng)
            doc_freq.update(set(ng))

        top_vocab = [w for w, _ in doc_freq.most_common(self.max_features)]
        self.vocab = {w: i for i, w in enumerate(top_vocab)}
        V = len(self.vocab)
        N = len(texts)

        df = np.zeros(V, dtype=np.float32)
        for ng in all_ng:
            for g in set(ng):
                if g in self.vocab:
                    df[self.vocab[g]] += 1
        self.idf = np.log((N + 1) / (df + 1)) + 1.0

        return self._build_matrix(all_ng)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._build_matrix([self._ngrams(t) for t in texts])

    def _build_matrix(self, all_ng: List[List[str]]) -> np.ndarray:
        V = len(self.vocab)
        X = np.zeros((len(all_ng), V), dtype=np.float32)
        for i, ng in enumerate(all_ng):
            c = Counter(ng)
            for g, cnt in c.items():
                if g in self.vocab:
                    X[i, self.vocab[g]] = cnt * self.idf[self.vocab[g]]
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return X / norms

def run_char_ngram_svm(texts_train: List[str], y_train: np.ndarray, texts_test: List[str], y_test: np.ndarray) -> Dict:
    print(f"  Fitting Char N-grams (max_features=15k, n=3-5)...")
    vec = CharNgramVectorizerNumpy(max_features=15000, ngram_range=(3, 5))
    X_tr = vec.fit_transform(texts_train)
    X_te = vec.transform(texts_test)
    
    print("  Training PyTorch SVM on Char N-grams...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = train_svm_pytorch(X_tr, y_train, X_te, y_test, epochs=300, lr=0.01, device=device)
    
    metrics.update({
        "baseline": "char_ngrams_svm",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "vocab_size": len(vec.vocab)
    })
    
    return metrics
