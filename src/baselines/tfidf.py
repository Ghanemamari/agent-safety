"""
TF-IDF Baseline (Numpy implementation without Sklearn)
"""
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import Dict, List
from src.eval.metrics import compute_metrics

class TfidfVectorizerNumpy:
    def __init__(self, max_features: int = 30000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = None

    def _ngrams(self, text: str) -> List[str]:
        tokens = text.lower().split()
        out = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                out.append(" ".join(tokens[i : i + n]))
        return out

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        doc_freq: Counter = Counter()
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

def run_tfidf_numpy(texts_tr, y_tr, texts_te, y_te, lr=0.01, epochs=200):
    vec = TfidfVectorizerNumpy()
    X_tr = vec.fit_transform(texts_tr)
    X_te = vec.transform(texts_te)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    
    model = nn.Linear(X_tr.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb).squeeze(-1), yb).backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te_t).squeeze(-1)).cpu().numpy()
        
    return compute_metrics(y_te, probs)

