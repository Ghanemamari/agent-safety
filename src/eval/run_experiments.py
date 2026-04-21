r"""
Final Benchmark Experiments
===========================
Executes the evaluation matrix across three fundamental blocks:
 - Block 1: In-domain (Train on A, Test on A 70/30 split)
 - Block 2: Cross-dataset (Train on A, Test on B)
 - Block 3: Leave-one-dataset-out (Train on All minus A, Test on A)

Usage:
    python -m src.eval.run_experiments \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        [--fast]   # limit to 100 samples per dataset
"""

import os
import json
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict

# ── project internals ──────────────────────────────────────────────────────────
from src.utils.io import read_jsonl
from scripts.run_all_baselines import torch_logreg, compute_metrics as _np_metrics
from src.baselines.control_baselines import run_random, run_length, run_labelflip
from src.baselines.tfidf import run_tfidf_numpy
from src.baselines.semantic_mlp import run_semantic_mlp, train_mlp_on_embeddings
from src.baselines.perplexity import run_perplexity
from src.baselines.llamaguard import run_llamaguard

# ── dataset registry ───────────────────────────────────────────────────────────
DATASET_JSONLS: Dict[str, str] = {
    "injecagent":  "data/raw/injecagent.jsonl",
    "advbench":    "data/raw/advbench.jsonl",
    "agentdojo":   "data/raw/agentdojo.jsonl",
    "stealthy":    "data/raw/prompts_stealthy.jsonl",
}

LLMLAT_BENIGN = "data/raw/llmlat_benign.jsonl"

# ── helpers ────────────────────────────────────────────────────────────────────

def label_to_int(val) -> int:
    s = str(val).lower()
    return 1 if s in ("1", "true", "unsafe", "injected", "malicious") else 0


def load_texts_labels(ds_name: str, limit: int = None):
    """Load texts + binary labels, replacing benign with LLM-LAT if available."""
    path = DATASET_JSONLS[ds_name]
    data = read_jsonl(path)
    if limit:
        data = data[:limit]

    mal_texts = [d["prompt"] for d in data if label_to_int(d["label"]) == 1]

    if os.path.exists(LLMLAT_BENIGN):
        lat_data = read_jsonl(LLMLAT_BENIGN)
        if limit:
            lat_data = lat_data[:limit]
        ben_texts = [d["prompt"] for d in lat_data][: len(mal_texts)]
    else:
        ben_texts = [d["prompt"] for d in data if label_to_int(d["label"]) == 0]

    texts = mal_texts + ben_texts
    labels = np.array([1] * len(mal_texts) + [0] * len(ben_texts), dtype=np.float32)
    return texts, labels


def collect_texts_labels(ds_list: List[str], limit: int = None):
    all_texts, all_labels = [], []
    for ds in ds_list:
        t, y = load_texts_labels(ds, limit)
        all_texts.extend(t)
        all_labels.extend(y)
    return all_texts, np.array(all_labels, dtype=np.float32)


# ── feature extraction via subprocess (memory-safe) ────────────────────────────

def _npz_path(model_name: str, ds_name: str, fast: bool = False) -> str:
    safe = model_name.replace("/", "_").replace(":", "_")
    suffix = "_fast" if fast else ""
    return f"data/features/{safe}_{ds_name}_feats{suffix}.npz"


def _extract_subprocess(model_name: str, input_jsonl: str, out_npz: str, load4bit: bool):
    cmd = [sys.executable, "-m", "src.extract.extract_activations",
           "--model", model_name, "--input", input_jsonl, "--out", out_npz]
    if load4bit:
        cmd.append("--load4bit")
    subprocess.run(cmd, check=True)


def ensure_features(model_name: str, load4bit: bool, fast: bool):
    os.makedirs("data/features", exist_ok=True)
    limit = 100 if fast else None

    for ds_name, path in DATASET_JSONLS.items():
        if not os.path.exists(path):
            continue
        npz = _npz_path(model_name, ds_name, fast)
        if not os.path.exists(npz):
            src = path
            if fast:
                src = f"data/raw/{ds_name}_fast.jsonl"
                rows = read_jsonl(path)[:limit]
                with open(src, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            print(f"  [extract] {ds_name}...")
            _extract_subprocess(model_name, src, npz, load4bit)

    # LLM-LAT benign features
    if os.path.exists(LLMLAT_BENIGN):
        npz_lat = _npz_path(model_name, "llmlat_benign", fast)
        if not os.path.exists(npz_lat):
            src = LLMLAT_BENIGN
            if fast:
                src = "data/raw/llmlat_benign_fast.jsonl"
                rows = read_jsonl(LLMLAT_BENIGN)[:limit]
                with open(src, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            print(f"  [extract] llmlat_benign...")
            _extract_subprocess(model_name, src, npz_lat, load4bit)


def load_features(npz: str):
    d = np.load(npz, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32)


def collect_features(model_name: str, ds_list: List[str], fast: bool):
    """Combine mal features from attack datasets with LLM-LAT benign features."""
    Xs, Ys = [], []
    npz_lat_path = _npz_path(model_name, "llmlat_benign", fast)
    X_lat, _ = load_features(npz_lat_path) if os.path.exists(npz_lat_path) else (None, None)

    for ds in ds_list:
        npz = _npz_path(model_name, ds, fast)
        if not os.path.exists(npz):
            print(f"  [warn] Missing features for {ds}, skipping.")
            continue
        X, y = load_features(npz)
        mal_idx = (y == 1)
        X_mal = X[mal_idx]

        if X_lat is not None:
            X_ben = X_lat[: len(X_mal)]
            X_comb = np.vstack([X_mal, X_ben])
            y_comb = np.array([1.0] * len(X_mal) + [0.0] * len(X_ben))
        else:
            X_comb = X
            y_comb = y

        Xs.append(X_comb)
        Ys.append(y_comb)

    return np.vstack(Xs), np.concatenate(Ys)


# ── split helpers ──────────────────────────────────────────────────────────────

def stratified_split_70_30(y: np.ndarray, seed: int = 42):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0].copy(); rng.shuffle(pos)
    neg = np.where(y == 0)[0].copy(); rng.shuffle(neg)
    n_pos_te = max(1, int(len(pos) * 0.3))
    n_neg_te = max(1, int(len(neg) * 0.3))
    tr = np.concatenate([pos[n_pos_te:], neg[n_neg_te:]])
    te = np.concatenate([pos[:n_pos_te], neg[:n_neg_te]])
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


# ── evaluation core ────────────────────────────────────────────────────────────

def safe_append(results: list, block: str, method: str, m: dict):
    results.append({
        "Block":    block,
        "Method":   method,
        "AUROC":    round(m.get("auroc",    float("nan")), 4),
        "F1-Score": round(m.get("f1",       float("nan")), 4),
        "Accuracy": round(m.get("accuracy", float("nan")), 4),
    })


def eval_all_baselines(
    X_tr, y_tr, X_te, y_te,
    t_tr: List[str], y_tr_txt: np.ndarray,
    t_te: List[str], y_te_txt: np.ndarray,
    tag: str,
    results: list,
    model_name_for_ppl: str,
    run_heavy: bool = False,
):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # -- normalise hidden-state features --
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_te_n = (X_te - mean) / std

    # 1. Linear Probe (on hidden states)
    print(f"  [{tag}] Linear Probe...")
    m = torch_logreg(X_tr_n, y_tr, X_te_n, y_te, epochs=200, lr=0.01, device=device)
    safe_append(results, tag, "Linear Probe", m)

    # 2. MLP Probe (on hidden states)
    print(f"  [{tag}] MLP Probe...")
    m = train_mlp_on_embeddings(X_tr_n, y_tr, X_te_n, y_te, epochs=100, device=device)
    safe_append(results, tag, "MLP Probe", m)

    # 3. TF-IDF + Logistic Regression (pure NumPy/Torch, no sklearn)
    print(f"  [{tag}] TF-IDF + LR...")
    m = run_tfidf_numpy(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, tag, "TF-IDF + LR", m)

    # 4. Semantic Embeddings MLP (SentenceTransformers)
    print(f"  [{tag}] Semantic Embeddings MLP...")
    m = run_semantic_mlp(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, tag, "Semantic MLP", m)

    # 5. Perplexity (causal LM loss as anomaly score)
    print(f"  [{tag}] Perplexity...")
    m = run_perplexity(t_tr, y_tr_txt, t_te, y_te_txt, model_name=model_name_for_ppl)
    safe_append(results, tag, "Perplexity", m)

    # 6. LlamaGuard 3 (optional – skip if not gated/too slow)
    if run_heavy:
        print(f"  [{tag}] LlamaGuard 3...")
        m = run_llamaguard(t_te, y_te_txt)
        safe_append(results, tag, "LlamaGuard 3", m)

    # -- Controls --
    # 7. Random
    m = run_random(y_te_txt)
    safe_append(results, tag, "Random", m)

    # 8. Length Classifier
    m = run_length(t_tr, y_tr_txt, t_te, y_te_txt)
    safe_append(results, tag, "Length Classifier", m)

    # 9. Label-Flip
    m = run_labelflip(X_tr_n, y_tr, X_te_n, y_te)
    safe_append(results, tag, "Label-Flip Control", m)


# ── orchestration ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fast", action="store_true",
                    help="Limit each dataset to 100 samples (debug speed)")
    ap.add_argument("--load4bit", action="store_true",
                    help="4-bit quantisation for feature extraction")
    ap.add_argument("--llamaguard", action="store_true",
                    help="Include LlamaGuard 3 inference (slow / memory heavy)")
    args = ap.parse_args()

    fast = args.fast
    model = args.model

    # ── 0. Extract hidden-state features (subprocess per dataset) ─────────────
    print("=" * 70)
    print(f"  Ensuring features exist  (fast={fast})")
    ensure_features(model, args.load4bit, fast)

    # ── which datasets are available? ─────────────────────────────────────────
    datasets = [name for name, path in DATASET_JSONLS.items() if os.path.exists(path)]
    print(f"\n  Available datasets: {datasets}")
    if len(datasets) < 2:
        print("Need at least 2 datasets. Exiting.")
        return

    results = []

    # ── BLOCK 1 : In-Domain (70 / 30 split per dataset) ──────────────────────
    print("\n" + "=" * 70)
    print("  BLOCK 1: In-Domain")
    for ds in datasets:
        tag = f"Block1 (In-Domain: {ds})"
        t, y_txt = load_texts_labels(ds, 100 if fast else None)
        X, y    = collect_features(model, [ds], fast)

        # align lengths (text vs feature array may differ due to balanced LLM-LAT)
        n = min(len(t), len(y))
        t = t[:n]; y_txt = y_txt[:n]; X = X[:n]; y = y[:n]

        tr_idx, te_idx = stratified_split_70_30(y)
        eval_all_baselines(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            [t[i] for i in tr_idx], y_txt[tr_idx],
            [t[i] for i in te_idx], y_txt[te_idx],
            tag, results, model, run_heavy=args.llamaguard,
        )

    # ── BLOCK 2 : Cross-Dataset (Train A → Test B) ────────────────────────────
    print("\n" + "=" * 70)
    print("  BLOCK 2: Cross-Dataset")
    for ds_tr in datasets:
        for ds_te in datasets:
            if ds_tr == ds_te:
                continue
            tag = f"Block2 (Train: {ds_tr} -> Test: {ds_te})"
            limit = 100 if fast else None
            t_tr, y_tr_txt = load_texts_labels(ds_tr, limit)
            t_te, y_te_txt = load_texts_labels(ds_te, limit)
            X_tr, y_tr = collect_features(model, [ds_tr], fast)
            X_te, y_te = collect_features(model, [ds_te], fast)
            n_tr = min(len(t_tr), len(y_tr))
            n_te = min(len(t_te), len(y_te))
            eval_all_baselines(
                X_tr[:n_tr], y_tr[:n_tr], X_te[:n_te], y_te[:n_te],
                t_tr[:n_tr], y_tr_txt[:n_tr], t_te[:n_te], y_te_txt[:n_te],
                tag, results, model, run_heavy=args.llamaguard,
            )

    # ── BLOCK 3 : Leave-One-Dataset-Out ──────────────────────────────────────
    if len(datasets) > 2:
        print("\n" + "=" * 70)
        print("  BLOCK 3: Leave-One-Dataset-Out")
        for target in datasets:
            train_ds = [d for d in datasets if d != target]
            tag = f"Block3 (LOO -> Test: {target})"
            limit = 100 if fast else None
            t_tr, y_tr_txt = collect_texts_labels(train_ds, limit)
            t_te, y_te_txt = load_texts_labels(target, limit)
            X_tr, y_tr = collect_features(model, train_ds, fast)
            X_te, y_te = collect_features(model, [target], fast)
            n_tr = min(len(t_tr), len(y_tr))
            n_te = min(len(t_te), len(y_te))
            eval_all_baselines(
                X_tr[:n_tr], y_tr[:n_tr], X_te[:n_te], y_te[:n_te],
                t_tr[:n_tr], y_tr_txt[:n_tr], t_te[:n_te], y_te_txt[:n_te],
                tag, results, model, run_heavy=args.llamaguard,
            )

    # ── Save & Print ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    os.makedirs("data/results", exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    out_csv = f"data/results/eval_blocks_{ts}.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS".center(70))
    print("=" * 70)
    # Pretty pivot table
    pivot = df.pivot_table(
        index="Method", columns="Block",
        values=["AUROC", "F1-Score", "Accuracy"], aggfunc="mean"
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(pivot.round(3))
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    main()
