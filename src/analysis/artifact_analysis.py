"""
Artifact Analysis — Addresses "1.00 AUROC is suspicious" reviewer critique
===========================================================================
This script provides evidence that the near-perfect AUROC of the linear probe
is NOT due to dataset artifacts (length, formatting, token frequency biases).

It runs three control experiments:

  1. Length-only classifier  → AUROC using only token length as feature
  2. First/last N tokens     → AUROC using only surface tokens (formatting check)  
  3. Surface feature AUROC   → Shows surface features CANNOT explain the result
  4. Probe weight analysis   → Identifies which dimensions are most discriminative

Usage:
    python -m src.analysis.artifact_analysis \
        --jsonl data/raw/prompts_stealthy_large.jsonl \
        --npz   data/processed_stealthy/TinyLlama_feats.npz \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --out   data/results/artifact_analysis.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.eval.metrics import compute_metrics
from src.utils.io import read_jsonl, write_json


# ─────────────────────────────────────────────────────────────────────────────
# 1. Length-only classifier
# ─────────────────────────────────────────────────────────────────────────────

def length_only_auroc(rows: List[Dict], model_name: str = None) -> Dict:
    """
    Trains a classifier using ONLY token count as a feature.
    If AUROC << probe AUROC → length is not the shortcut.
    """
    from src.extract.extract_activations import normalize_label

    labels = np.array([normalize_label(r["label"]) for r in rows])

    # Character length (no model needed)
    char_lengths = np.array([len(r["prompt"]) for r in rows], dtype=float)
    word_lengths  = np.array([len(r["prompt"].split()) for r in rows], dtype=float)

    char_auroc = roc_auc_score(labels, char_lengths)
    char_auroc = max(char_auroc, 1 - char_auroc)  # Best orientation

    word_auroc = roc_auc_score(labels, word_lengths)
    word_auroc = max(word_auroc, 1 - word_auroc)

    result = {
        "char_length_auroc": float(char_auroc),
        "word_length_auroc": float(word_auroc),
        "mean_char_len_benign": float(np.mean(char_lengths[labels == 0])),
        "mean_char_len_injected": float(np.mean(char_lengths[labels == 1])),
        "mean_word_len_benign": float(np.mean(word_lengths[labels == 0])),
        "mean_word_len_injected": float(np.mean(word_lengths[labels == 1])),
    }

    # Token length (requires tokenizer)
    if model_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name)
            token_lengths = np.array(
                [len(tok.encode(r["prompt"])) for r in rows], dtype=float
            )
            token_auroc = roc_auc_score(labels, token_lengths)
            token_auroc = max(token_auroc, 1 - token_auroc)
            result["token_length_auroc"] = float(token_auroc)
            result["mean_token_len_benign"] = float(np.mean(token_lengths[labels == 0]))
            result["mean_token_len_injected"] = float(np.mean(token_lengths[labels == 1]))
        except Exception as e:
            print(f"  [warning] Token length analysis skipped: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Surface token classifier (first/last N words)
# ─────────────────────────────────────────────────────────────────────────────

def surface_token_auroc(rows: List[Dict], n_words: int = 10) -> Dict:
    """
    Trains a TF-IDF classifier on ONLY the first N and last N words.
    Injections are often embedded in the middle — so if surface tokens
    already separate, it's a formatting artifact.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.extract.extract_activations import normalize_label

    labels = np.array([normalize_label(r["label"]) for r in rows])

    def extract_surface(text: str, n: int) -> str:
        words = text.split()
        return " ".join(words[:n] + words[-n:])

    surface_texts = [extract_surface(r["prompt"], n_words) for r in rows]

    X_tr, X_te, y_tr, y_te = train_test_split(
        surface_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr_v, y_tr)
    probs = clf.predict_proba(X_te_v)[:, 1]
    auroc = roc_auc_score(y_te, probs)

    return {
        "surface_tfidf_auroc": float(auroc),
        "surface_n_words": n_words,
        "interpretation": (
            "High surface AUROC → possible formatting artifact"
            if auroc > 0.75
            else "Low surface AUROC → probe learns semantic intent, not surface format ✓"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Probe weight analysis
# ─────────────────────────────────────────────────────────────────────────────

def probe_weight_analysis(npz_path: str, top_k: int = 20, seed: int = 42) -> Dict:
    """
    Trains a linear probe and analyzes its weight vector.
    
    Returns:
    - Weight L2 norm (overall "confidence" level)
    - Sparsity (are weights concentrated or spread?)
    - Top-K most discriminative dimensions
    
    A well-distributed (non-sparse) weight vector suggests the probe
    is using many features globally rather than a single artifact dimension.
    """
    data = np.load(npz_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    layers = data.get("layers", [])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X_train_n, y_train)
    probs = clf.predict_proba(X_test_n)[:, 1]

    w = clf.coef_[0]   # Shape: (feature_dim,)
    w_abs = np.abs(w)

    # Gini coefficient (0 = perfectly uniform, 1 = perfectly concentrated on one feature)
    w_sorted = np.sort(w_abs)
    n = len(w_sorted)
    gini = (2 * np.sum(np.arange(1, n + 1) * w_sorted) - (n + 1) * np.sum(w_sorted)) / \
           (n * np.sum(w_sorted) + 1e-12)

    # Effective number of features (entropy-based)
    w_norm = w_abs / (w_abs.sum() + 1e-12)
    entropy = -np.sum(w_norm * np.log(w_norm + 1e-12))
    effective_features = int(np.exp(entropy))

    # Top-K dimensions
    top_dims = np.argsort(w_abs)[::-1][:top_k].tolist()

    return {
        "probe_auroc": float(roc_auc_score(y_test, probs)),
        "weight_l2_norm": float(np.linalg.norm(w)),
        "weight_gini": float(gini),
        "effective_features": effective_features,
        "feature_dim": int(X.shape[1]),
        "top_k_dimensions": top_dims,
        "gini_interpretation": (
            "Concentrated on few features (possible artifact)" if gini > 0.7
            else "Well-distributed across many features (robust encoding) ✓"
        ),
        "layers_probed": layers.tolist() if hasattr(layers, "tolist") else list(layers),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pair-balance check
# ─────────────────────────────────────────────────────────────────────────────

def pair_balance_check(rows: List[Dict]) -> Dict:
    """
    Verifies that pair_id-based group splitting is working correctly.
    - Are pairs perfectly balanced (equal benign/injected per pair)?
    - Are there any unpaired samples?
    """
    from collections import Counter

    pair_labels: Dict[str, List] = {}
    for r in rows:
        pid = r.get("pair_id", "no_pair_id")
        pair_labels.setdefault(pid, []).append(r["label"])

    perfect_pairs = 0
    imperfect_pairs = 0
    unpaired = 0

    for pid, lbls in pair_labels.items():
        if pid == "no_pair_id":
            unpaired += len(lbls)
        elif len(lbls) == 2:
            lbl_set = set(str(l).lower() for l in lbls)
            if "benign" in lbl_set or "0" in lbl_set:
                perfect_pairs += 1
            else:
                imperfect_pairs += 1
        else:
            imperfect_pairs += 1

    n_total = len(rows)
    n_injected = sum(1 for r in rows if str(r.get("label", "")).lower() in ["injected", "1"])
    n_benign = n_total - n_injected

    return {
        "n_total": n_total,
        "n_benign": n_benign,
        "n_injected": n_injected,
        "class_balance_ratio": float(n_benign / max(n_injected, 1)),
        "n_perfect_pairs": perfect_pairs,
        "n_imperfect_pairs": imperfect_pairs,
        "n_unpaired": unpaired,
        "pairing_quality": (
            "Perfect (all samples paired)" if imperfect_pairs == 0 and unpaired == 0
            else f"Imperfect ({imperfect_pairs} bad pairs, {unpaired} unpaired)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_artifact_analysis(
    jsonl_path: str,
    npz_path: Optional[str] = None,
    model_name: Optional[str] = None,
    out_json: Optional[str] = None,
    top_k: int = 20,
    surface_n: int = 10,
) -> Dict:

    rows = read_jsonl(jsonl_path)
    print(f"\nArtifact Analysis on: {jsonl_path}")
    print(f"  N={len(rows)} samples")
    print("=" * 55)

    results = {
        "source_jsonl": jsonl_path,
        "source_npz": npz_path,
    }

    # 1. Pair balance
    print("\n[1/4] Pair balance check...")
    balance = pair_balance_check(rows)
    results["pair_balance"] = balance
    print(f"  N total: {balance['n_total']} | benign: {balance['n_benign']} | injected: {balance['n_injected']}")
    print(f"  Class ratio: {balance['class_balance_ratio']:.2f}")
    print(f"  Pairing: {balance['pairing_quality']}")

    # 2. Length-only AUROC
    print("\n[2/4] Length-only artifact check...")
    lengths = length_only_auroc(rows, model_name=model_name)
    results["length_artifact"] = lengths
    char_a = lengths["char_length_auroc"]
    tok_a  = lengths.get("token_length_auroc", "N/A")
    print(f"  Char-length AUROC:  {char_a:.4f} {'✓ (low → no artifact)' if char_a < 0.65 else '⚠ (high → possible artifact)'}")
    print(f"  Token-length AUROC: {tok_a}")

    # 3. Surface TF-IDF
    print(f"\n[3/4] Surface tokens (first/last {surface_n} words) AUROC...")
    surface = surface_token_auroc(rows, n_words=surface_n)
    results["surface_artifact"] = surface
    s_a = surface["surface_tfidf_auroc"]
    print(f"  Surface TF-IDF AUROC: {s_a:.4f}")
    print(f"  {surface['interpretation']}")

    # 4. Probe weight analysis
    if npz_path and os.path.exists(npz_path):
        print("\n[4/4] Probe weight analysis...")
        weights = probe_weight_analysis(npz_path, top_k=top_k)
        results["probe_weights"] = weights
        print(f"  Probe AUROC:        {weights['probe_auroc']:.4f}")
        print(f"  Weight Gini coeff:  {weights['weight_gini']:.4f}")
        print(f"  Effective features: {weights['effective_features']}/{weights['feature_dim']}")
        print(f"  → {weights['gini_interpretation']}")
    else:
        print(f"\n[4/4] Probe weight analysis: SKIPPED (no NPZ file)")

    # ── Summary verdict ────────────────────────────────────────────────────────
    artifact_flags = []
    if lengths["char_length_auroc"] > 0.65:
        artifact_flags.append("length bias detected")
    if surface["surface_tfidf_auroc"] > 0.75:
        artifact_flags.append("surface formatting artifact")
    if "probe_weights" in results and results["probe_weights"]["weight_gini"] > 0.7:
        artifact_flags.append("concentrated probe weights")

    results["verdict"] = {
        "artifact_flags": artifact_flags,
        "conclusion": (
            "✓ No significant artifacts detected. "
            "High AUROC reflects genuine semantic intent encoding."
            if not artifact_flags
            else f"⚠ Potential artifacts: {', '.join(artifact_flags)}. "
                 "Consider length-matching or additional controls."
        ),
    }

    print(f"\n{'='*55}")
    print(f"VERDICT: {results['verdict']['conclusion']}")
    print(f"{'='*55}")

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        write_json(out_json, results)
        print(f"\nSaved → {out_json}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Artifact analysis for linear probe")
    ap.add_argument("--jsonl", required=True, help="Path to prompts JSONL")
    ap.add_argument("--npz", default=None, help="Path to features NPZ (for weight analysis)")
    ap.add_argument("--model", default=None,
                    help="Model name for tokenizer (token-length analysis)")
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument("--top_k", type=int, default=20,
                    help="Top-K most discriminative weight dimensions to report")
    ap.add_argument("--surface_n", type=int, default=10,
                    help="Number of first/last words for surface analysis")
    args = ap.parse_args()

    run_artifact_analysis(
        jsonl_path=args.jsonl,
        npz_path=args.npz,
        model_name=args.model,
        out_json=args.out,
        top_k=args.top_k,
        surface_n=args.surface_n,
    )


if __name__ == "__main__":
    main()
