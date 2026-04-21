"""
run_all_baselines.py
====================
Runs TF-IDF, Semantic, and MLP probe baselines on InjecAgent & AdvBench.

Implemented in PURE PyTorch + NumPy — no sklearn/scipy dependency,
to bypass Windows AppLocker DLL restrictions.

Usage:
    python scripts/run_all_baselines.py                         # all baselines, all datasets
    python scripts/run_all_baselines.py --baselines tfidf semantic
    python scripts/run_all_baselines.py --datasets injecagent
"""

import io
import os
import sys
import json
import argparse
import glob
from collections import Counter
from typing import Dict, List

# Force UTF-8 output on Windows (avoids CP1252 UnicodeEncodeError)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

# ── ROC-AUC + metrics (pure NumPy) ────────────────────────────────────────────

def roc_auc_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_s = y_true[desc_idx]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(_trapz(tpr, fpr))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(float)
    auroc = roc_auc_numpy(y_true, probs)
    acc = float((preds == y_true).mean())
    tp = float(((preds == 1) & (y_true == 1)).sum())
    fp = float(((preds == 1) & (y_true == 0)).sum())
    fn = float(((preds == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "auroc": round(auroc, 6),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_to_int(val) -> int:
    s = str(val).lower()
    return 1 if s in ["1", "true", "unsafe", "malicious", "injected"] else 0


def stratified_split(y: np.ndarray, test_size: float = 0.3, seed: int = 42):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0].copy(); rng.shuffle(pos_idx)
    neg_idx = np.where(y == 0)[0].copy(); rng.shuffle(neg_idx)
    n_pos_te = max(1, int(len(pos_idx) * test_size))
    n_neg_te = max(1, int(len(neg_idx) * test_size))
    tr_idx = np.concatenate([pos_idx[n_pos_te:], neg_idx[n_neg_te:]])
    te_idx = np.concatenate([pos_idx[:n_pos_te], neg_idx[:n_neg_te]])
    rng.shuffle(tr_idx); rng.shuffle(te_idx)
    return tr_idx, te_idx


def save_json(path: str, data: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── PyTorch Logistic Regression ────────────────────────────────────────────────

def torch_logreg(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    lr: float = 0.01,
    epochs: int = 400,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
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
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb).squeeze(-1), yb).backward()
            optimizer.step()
        if verbose and (ep + 1) % 100 == 0:
            print(f"    epoch {ep+1}/{epochs}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te_t).squeeze(-1)).cpu().numpy()
    return compute_metrics(y_te, probs)


# ── TF-IDF (pure Python + NumPy) ──────────────────────────────────────────────

class TfidfVectorizerNumpy:
    """Lightweight TF-IDF with (1,2)-gram support — no sklearn dependency."""

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
        self.idf = np.log((N + 1) / (df + 1)) + 1.0  # smooth IDF

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


# ── MLP Architecture ───────────────────────────────────────────────────────────

class IntentMLP(nn.Module):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 1 — TF-IDF + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

def run_tfidf(input_path: str, out_path: str, seed: int = 42) -> Dict:
    print("\n" + "="*62)
    print(f"[TF-IDF] {os.path.basename(input_path)}")
    rows = load_jsonl(input_path)
    texts = [r["prompt"] for r in rows]
    y = np.array([label_to_int(r["label"]) for r in rows])
    print(f"  Samples: {len(y)}  |  Injected: {int(y.sum())}  |  Benign: {int((y==0).sum())}")

    tr_idx, te_idx = stratified_split(y, test_size=0.3, seed=seed)
    texts_tr, texts_te = [texts[i] for i in tr_idx], [texts[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"  Fitting TF-IDF (max_features=30k, ngrams=1-2)...")
    vec = TfidfVectorizerNumpy(max_features=30000, ngram_range=(1, 2))
    X_tr = vec.fit_transform(texts_tr)
    X_te = vec.transform(texts_te)
    print(f"  Vocab: {len(vec.vocab)}  |  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    print("  Training PyTorch logistic regression...")
    metrics = torch_logreg(X_tr, y_tr, X_te, y_te, lr=0.05, epochs=400, verbose=True)
    metrics.update(
        baseline="tfidf_logreg",
        dataset=os.path.basename(input_path),
        n_train=int(len(y_tr)),
        n_test=int(len(y_te)),
        vocab_size=len(vec.vocab),
    )

    print(f"  >> AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")
    save_json(out_path, metrics)
    print(f"  Saved -> {out_path}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 2 — Semantic Embeddings + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

def run_semantic(
    input_path: str,
    out_path: str,
    embed_model: str = "all-MiniLM-L6-v2",
    seed: int = 42,
) -> Dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [SKIP] sentence_transformers not installed.")
        return {}

    print("\n" + "="*62)
    print(f"[Semantic] {os.path.basename(input_path)}  |  model: {embed_model}")
    rows = load_jsonl(input_path)
    prompts = [r["prompt"] for r in rows]
    y = np.array([label_to_int(r["label"]) for r in rows])
    print(f"  Samples: {len(y)}  |  Injected: {int(y.sum())}  |  Benign: {int((y==0).sum())}")

    print(f"  Loading encoder...")
    encoder = SentenceTransformer(embed_model)
    print(f"  Encoding {len(prompts)} prompts (batch=128)...")
    X = encoder.encode(prompts, show_progress_bar=True, batch_size=128)

    tr_idx, te_idx = stratified_split(y, test_size=0.3, seed=seed)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Z-score normalise
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    print(f"  Embedding dim: {X.shape[1]}  |  Train: {len(y_tr)}  |  Test: {len(y_te)}")
    print("  Training logistic regression on embeddings...")
    metrics = torch_logreg(X_tr, y_tr, X_te, y_te, lr=0.01, epochs=500, verbose=True)
    metrics.update(
        baseline="semantic_logreg",
        embedding_model=embed_model,
        dataset=os.path.basename(input_path),
        n_train=int(len(y_tr)),
        n_test=int(len(y_te)),
        embed_dim=int(X.shape[1]),
    )

    print(f"  >> AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")
    save_json(out_path, metrics)
    print(f"  Saved -> {out_path}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 3 — MLP Probe on hidden-state NPZ activations
# ═══════════════════════════════════════════════════════════════════════════════

def run_mlp_probe(
    npz_path: str,
    out_path: str,
    seed: int = 42,
    epochs: int = 100,
    compare_linear: bool = True,
) -> Dict:
    print("\n" + "="*62)
    print(f"[MLP Probe] {os.path.relpath(npz_path)}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    model_name = str(data["model_name"][0]) if "model_name" in data else "unknown"
    print(f"  Model: {model_name}  |  X: {X.shape}  |  Injected: {int(y.sum())}/{len(y)}")

    tr_idx, te_idx = stratified_split(y, test_size=0.3, seed=seed)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_te_n = (X_te - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tr_t = torch.tensor(X_tr_n).to(device)
    y_tr_t = torch.tensor(y_tr).to(device)
    X_te_t = torch.tensor(X_te_n).to(device)

    n_pos = float(y_tr.sum()); n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

    mlp = IntentMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    best_loss = float("inf"); best_state = None
    mlp.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(mlp(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        ep_loss /= max(len(y_tr), 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        if (ep + 1) % 25 == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={ep_loss:.4f}")

    if best_state:
        mlp.load_state_dict(best_state)

    mlp.eval()
    with torch.no_grad():
        mlp_probs = torch.sigmoid(mlp(X_te_t)).cpu().numpy()

    mlp_metrics = compute_metrics(y_te, mlp_probs)
    print(f"  MLP  >> AUROC={mlp_metrics['auroc']:.4f}  F1={mlp_metrics['f1']:.4f}")

    if compare_linear:
        lin_m = torch_logreg(X_tr_n, y_tr, X_te_n, y_te, epochs=300, lr=0.01, device=device)
        print(f"  Lin  >> AUROC={lin_m['auroc']:.4f}  F1={lin_m['f1']:.4f}")
        delta = mlp_metrics["auroc"] - lin_m["auroc"]
        tag = "LINEARLY SEPARABLE [OK]" if abs(delta) < 0.02 else "non-linear structure present"
        print(f"  Delta AUROC (MLP-Lin): {delta:+.4f}  ->  {tag}")
        mlp_metrics["linear_auroc"] = lin_m["auroc"]
        mlp_metrics["delta_auroc_mlp_vs_linear"] = round(float(delta), 6)

    mlp_metrics.update(
        baseline="mlp_probe",
        model_name=model_name,
        npz=npz_path,
        n_train=int(len(y_tr)),
        n_test=int(len(y_te)),
        feature_dim=int(X.shape[1]),
        epochs=epochs,
        device=device,
    )

    save_json(out_path, mlp_metrics)
    print(f"  Saved -> {out_path}")
    return mlp_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATHS = {
    "injecagent": "data/raw/injecagent.jsonl",
    "advbench":   "data/raw/advbench.jsonl",
}


def main():
    ap = argparse.ArgumentParser(description="Run all baselines — no sklearn/scipy required")
    ap.add_argument(
        "--baselines", nargs="+",
        default=["tfidf", "semantic", "mlp"],
        choices=["tfidf", "semantic", "mlp", "all"],
        help="Which baselines to run",
    )
    ap.add_argument(
        "--datasets", nargs="+",
        default=["injecagent", "advbench"],
        choices=["injecagent", "advbench", "all"],
        help="Which datasets to evaluate on",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlp_epochs", type=int, default=100)
    args = ap.parse_args()

    baselines = set(args.baselines)
    if "all" in baselines:
        baselines = {"tfidf", "semantic", "mlp"}

    datasets = list(args.datasets)
    if "all" in datasets:
        datasets = list(DATASET_PATHS.keys())

    all_results: Dict = {}

    # ── TF-IDF & Semantic on raw datasets ────────────────────────────────────
    for ds in datasets:
        path = DATASET_PATHS[ds]
        if not os.path.exists(path):
            print(f"[WARN] Dataset not found: {path}")
            continue

        if "tfidf" in baselines:
            out = f"data/baselines/{ds}/tfidf_metrics.json"
            r = run_tfidf(path, out, seed=args.seed)
            if r:
                all_results[f"tfidf_{ds}"] = r

        if "semantic" in baselines:
            out = f"data/baselines/{ds}/semantic_metrics.json"
            r = run_semantic(path, out, seed=args.seed)
            if r:
                all_results[f"semantic_{ds}"] = r

    # ── MLP Probe on all available NPZ activations ────────────────────────────
    if "mlp" in baselines:
        npz_files = sorted(glob.glob("data/processed*/*.npz"))
        if not npz_files:
            print("\n[WARN] No NPZ files found under data/processed*/")
        for npz in npz_files:
            out_dir = os.path.dirname(npz)
            tag = os.path.splitext(os.path.basename(npz))[0]
            out = f"{out_dir}/{tag}_mlp_metrics.json"
            r = run_mlp_probe(npz, out, seed=args.seed, epochs=args.mlp_epochs)
            if r:
                all_results[f"mlp_{tag}"] = r

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_results:
        print("\n[WARN] No results collected. Check that datasets and NPZ files exist.")
        return

    print("\n" + "="*62)
    print(" FINAL SUMMARY")
    print("="*62)
    print(f"{'Baseline':<45} {'AUROC':>7} {'F1':>7} {'Acc':>7}")
    print("-" * 66)
    for name, m in sorted(all_results.items()):
        print(
            f"{name:<45} {m['auroc']:>7.4f} {m['f1']:>7.4f} {m['accuracy']:>7.4f}"
        )

    out_summary = "data/baselines/summary.json"
    save_json(out_summary, all_results)
    print(f"\nFull results -> {out_summary}")


if __name__ == "__main__":
    main()
