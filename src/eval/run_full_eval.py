"""
Full Evaluation Pipeline — Extended Baselines & Datasets
=========================================================
Runs the complete evaluation matrix:

  Datasets  : InjecAgent | AgentDojo | AdvBench | [your existing synthetic sets]
  Baselines : Linear Probe | MLP Probe | TF-IDF | Perplexity | Llama Guard

Usage (full run):
    python -m src.eval.run_full_eval \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --load4bit

Usage (specific dataset):
    python -m src.eval.run_full_eval \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --datasets injecagent advbench \
        --baselines linear mlp tfidf

Results saved to:  data/results/full_eval_results.json
                   data/results/full_eval_summary.csv
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Available datasets ─────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "injecagent": {
        "jsonl": "data/raw/injecagent.jsonl",
        "loader": "src.datasets.load_injecagent",
        "loader_args": ["--download"],
        "description": "InjecAgent (ACL 2024) — indirect prompt injection in tool-integrated agents",
        "ood": True,  # Used for OOD evaluation
    },
    "agentdojo": {
        "jsonl": "data/raw/agentdojo.jsonl",
        "loader": "src.datasets.load_agentdojo",
        "loader_args": [],
        "description": "AgentDojo (NeurIPS 2024) — realistic multi-domain agent injection tasks",
        "ood": True,
    },
    "advbench": {
        "jsonl": "data/raw/advbench.jsonl",
        "loader": "src.datasets.load_advbench",
        "loader_args": ["--download", "--subset", "behaviors"],
        "description": "AdvBench (Zou et al., 2023) — adversarial jailbreak behaviors",
        "ood": True,
    },
    # Existing synthetic datasets
    "stealthy": {
        "jsonl": "data/raw/prompts_stealthy.jsonl",
        "loader": None,
        "description": "Stealthy synthetic dataset (existing)",
        "ood": False,
    },
    "hard": {
        "jsonl": "data/raw/prompts_paired_hard.jsonl",
        "loader": None,
        "description": "Hard synthetic dataset (existing)",
        "ood": False,
    },
    "complex": {
        "jsonl": "data/raw/prompts_complex.jsonl",
        "loader": None,
        "description": "Complex synthetic dataset (existing)",
        "ood": False,
    },
}

# ── Available baselines ────────────────────────────────────────────────────────
BASELINE_REGISTRY = {
    "linear": {
        "description": "Linear Probe (Logistic Regression on hidden states)",
        "requires_activation": True,
    },
    "mlp": {
        "description": "MLP Non-linear Probe (ablation — 2-layer MLP on hidden states)",
        "requires_activation": True,
    },
    "tfidf": {
        "description": "TF-IDF + Logistic Regression (surface baseline)",
        "requires_activation": False,
    },
    "perplexity": {
        "description": "Perplexity-based detector (statistical baseline)",
        "requires_activation": False,
    },
    "llama_guard": {
        "description": "Llama Guard 3 (industrial safety classifier, OOD reference)",
        "requires_activation": False,
    },
}


def _run_dataset_loader(dataset_name: str, dataset_info: Dict) -> bool:
    """Download/prepare a dataset if its JSONL doesn't exist yet."""
    jsonl_path = dataset_info["jsonl"]
    if os.path.exists(jsonl_path):
        print(f"  [skip] {jsonl_path} already exists")
        return True

    if dataset_info.get("loader") is None:
        print(f"  [missing] {jsonl_path} — please generate this dataset first")
        return False

    cmd = [sys.executable, "-m", dataset_info["loader"]] + dataset_info.get("loader_args", []) + \
          ["--out", jsonl_path]

    print(f"  Preparing {dataset_name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_tfidf_baseline(jsonl_path: str, dataset_name: str) -> Dict:
    """Run TF-IDF baseline on a dataset."""
    from src.baselines.text_tfidf import load_jsonl, labels_to_int, train_and_eval
    from sklearn.model_selection import train_test_split

    rows = load_jsonl(jsonl_path)
    texts = [r["prompt"] for r in rows]
    y = labels_to_int([r["label"] for r in rows])

    X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.3, random_state=42, stratify=y)
    metrics = train_and_eval(X_tr, y_tr, X_te, y_te)
    metrics.update({
        "baseline": "tfidf",
        "dataset": dataset_name,
        "n_train": len(y_tr),
        "n_test": len(y_te),
    })
    return metrics


def run_perplexity_baseline(jsonl_path: str, dataset_name: str, model_name: str,
                             load4bit: bool = False) -> Dict:
    """Run perplexity-based baseline."""
    from src.baselines.statistical import calculate_perplexity
    from src.utils.io import read_jsonl
    from src.extract.extract_activations import normalize_label
    from sklearn.metrics import roc_auc_score
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"    Loading model for perplexity: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if load4bit:
        kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    rows = read_jsonl(jsonl_path)
    ppls, labels = [], []
    for r in rows:
        ppl, _ = calculate_perplexity(model, tokenizer, r["prompt"])
        ppls.append(ppl)
        labels.append(normalize_label(r["label"]))

    ppls_arr = np.array(ppls)
    labels_arr = np.array(labels)
    auroc = roc_auc_score(labels_arr, ppls_arr)

    # Test both orientations (high ppl = injected OR low ppl = injected)
    auroc_inv = roc_auc_score(labels_arr, -ppls_arr)
    best_auroc = max(auroc, auroc_inv)

    from src.eval.metrics import compute_metrics
    # Normalize PPL scores to [0,1] range
    ppl_min, ppl_max = ppls_arr.min(), ppls_arr.max()
    probs = (ppls_arr - ppl_min) / (ppl_max - ppl_min + 1e-8)
    metrics = compute_metrics(labels_arr, probs)
    metrics.update({
        "baseline": "perplexity",
        "dataset": dataset_name,
        "model_name": model_name,
        "auroc_raw": float(auroc),
        "auroc_inverted": float(auroc_inv),
        "auroc_best": float(best_auroc),
        "mean_ppl_benign": float(np.mean(ppls_arr[labels_arr == 0])),
        "mean_ppl_injected": float(np.mean(ppls_arr[labels_arr == 1])),
    })
    return metrics


def run_linear_and_mlp_probes(
    npz_path: str,
    dataset_name: str,
    compare_mlp: bool = True,
    seed: int = 42,
) -> Dict:
    """Run both linear and MLP probes on extracted features."""
    from src.probes.train_linear_probe import train_probe
    from src.baselines.mlp_probe import evaluate_mlp_probe

    all_metrics = {}

    # Linear probe
    linear_metrics = train_probe(npz_path, seed=seed)
    linear_metrics["baseline"] = "linear_probe"
    linear_metrics["dataset"] = dataset_name
    all_metrics["linear"] = linear_metrics

    # MLP probe (ablation)
    if compare_mlp:
        mlp_out = npz_path.replace("_feats.npz", "_mlp_metrics.json")
        mlp_metrics = evaluate_mlp_probe(
            npz_path=npz_path,
            out_json=mlp_out,
            seed=seed,
            compare_linear=False,
        )
        mlp_metrics["baseline"] = "mlp_probe"
        mlp_metrics["dataset"] = dataset_name
        all_metrics["mlp"] = mlp_metrics

    return all_metrics


def run_llama_guard_baseline(jsonl_path: str, dataset_name: str,
                              model_name: str = "meta-llama/Llama-Guard-3-1B",
                              load4bit: bool = False) -> Dict:
    """Run Llama Guard baseline."""
    from src.baselines.llama_guard import evaluate_llama_guard
    metrics = evaluate_llama_guard(
        input_jsonl=jsonl_path,
        model_name=model_name,
        load_in_4bit=load4bit,
    )
    metrics["dataset"] = dataset_name
    return metrics


def extract_features_for_dataset(
    model_name: str,
    jsonl_path: str,
    dataset_name: str,
    outdir: str,
    load4bit: bool = False,
) -> Optional[str]:
    """Extract hidden state features and return path to NPZ file."""
    from src.extract.extract_activations import extract_features

    safe_model = model_name.replace("/", "_").replace(":", "_")
    safe_dataset = dataset_name.replace("/", "_")
    npz_path = os.path.join(outdir, f"{safe_model}_{safe_dataset}_feats.npz")

    if os.path.exists(npz_path):
        print(f"    [cache] Features already exist: {npz_path}")
        return npz_path

    os.makedirs(outdir, exist_ok=True)
    try:
        extract_features(
            model_name=model_name,
            input_jsonl=jsonl_path,
            out_npz=npz_path,
            load_in_4bit=load4bit,
        )
        return npz_path
    except Exception as e:
        print(f"    [error] Feature extraction failed: {e}")
        return None


def run_full_evaluation(
    model_name: str,
    datasets: List[str],
    baselines: List[str],
    outdir: str = "data/results",
    load4bit: bool = False,
    llama_guard_model: str = "meta-llama/Llama-Guard-3-1B",
) -> Dict:
    """
    Main evaluation loop — runs all combinations of datasets × baselines.
    Returns a nested dict: {dataset_name: {baseline_name: metrics}}.
    """
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION PIPELINE")
    print(f"Model    : {model_name}")
    print(f"Datasets : {datasets}")
    print(f"Baselines: {baselines}")
    print(f"{'='*60}\n")

    os.makedirs(outdir, exist_ok=True)
    results = {}
    features_dir = os.path.join(outdir, "features")

    # ── Determine if we need to extract features ────────────────────────────────
    activation_baselines = [b for b in baselines if BASELINE_REGISTRY[b]["requires_activation"]]
    surface_baselines = [b for b in baselines if not BASELINE_REGISTRY[b]["requires_activation"]]

    for dataset_name in datasets:
        if dataset_name not in DATASET_REGISTRY:
            print(f"[skip] Unknown dataset: {dataset_name}")
            continue

        dataset_info = DATASET_REGISTRY[dataset_name]
        print(f"\n{'─'*50}")
        print(f"Dataset: {dataset_name} — {dataset_info['description']}")
        print(f"{'─'*50}")

        # Prepare dataset
        ok = _run_dataset_loader(dataset_name, dataset_info)
        if not ok:
            print(f"  [skip] Dataset {dataset_name} not available")
            continue

        jsonl_path = dataset_info["jsonl"]
        results[dataset_name] = {"_meta": dataset_info["description"]}

        # ── Surface baselines (no model needed) ─────────────────────────────────
        if "tfidf" in surface_baselines:
            print(f"\n  Running: TF-IDF baseline on {dataset_name}")
            try:
                m = run_tfidf_baseline(jsonl_path, dataset_name)
                results[dataset_name]["tfidf"] = m
                print(f"    AUROC={m.get('auroc', 'N/A'):.4f}")
            except Exception as e:
                print(f"    [error] TF-IDF failed: {e}")

        if "perplexity" in surface_baselines:
            print(f"\n  Running: Perplexity baseline on {dataset_name}")
            try:
                m = run_perplexity_baseline(jsonl_path, dataset_name, model_name, load4bit)
                results[dataset_name]["perplexity"] = m
                print(f"    AUROC (best)={m.get('auroc_best', 'N/A'):.4f}")
            except Exception as e:
                print(f"    [error] Perplexity failed: {e}")

        if "llama_guard" in surface_baselines:
            print(f"\n  Running: Llama Guard baseline on {dataset_name}")
            try:
                m = run_llama_guard_baseline(jsonl_path, dataset_name,
                                              model_name=llama_guard_model,
                                              load4bit=load4bit)
                results[dataset_name]["llama_guard"] = m
                print(f"    AUROC={m.get('auroc', 'N/A'):.4f}")
            except Exception as e:
                print(f"    [error] Llama Guard failed: {e}")

        # ── Activation-based baselines ──────────────────────────────────────────
        if activation_baselines:
            print(f"\n  Extracting hidden states: {model_name} on {dataset_name}")
            npz_path = extract_features_for_dataset(
                model_name=model_name,
                jsonl_path=jsonl_path,
                dataset_name=dataset_name,
                outdir=features_dir,
                load4bit=load4bit,
            )

            if npz_path:
                probe_results = run_linear_and_mlp_probes(
                    npz_path=npz_path,
                    dataset_name=dataset_name,
                    compare_mlp="mlp" in activation_baselines,
                )
                results[dataset_name].update(probe_results)

                if "linear" in probe_results:
                    print(f"    Linear Probe AUROC = {probe_results['linear'].get('auroc', 'N/A'):.4f}")
                if "mlp" in probe_results:
                    print(f"    MLP Probe    AUROC = {probe_results['mlp'].get('auroc', 'N/A'):.4f}")

    return results


def save_results(results: Dict, outdir: str):
    """Save results as JSON and as a summary CSV table."""
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON
    json_path = os.path.join(outdir, f"full_eval_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results → {json_path}")

    # Summary CSV (dataset × baseline → AUROC)
    rows = []
    for dataset, baseline_results in results.items():
        for baseline, metrics in baseline_results.items():
            if baseline.startswith("_"):
                continue
            if isinstance(metrics, dict) and "auroc" in metrics:
                rows.append({
                    "dataset": dataset,
                    "baseline": baseline,
                    "auroc": round(metrics.get("auroc", float("nan")), 4),
                    "f1": round(metrics.get("f1", float("nan")), 4),
                    "accuracy": round(metrics.get("accuracy", float("nan")), 4),
                    "n_test": metrics.get("n_test", ""),
                })

    if rows:
        df = pd.DataFrame(rows)
        # Pivot table: datasets as rows, baselines as columns
        pivot = df.pivot_table(
            index="dataset", columns="baseline", values="auroc",
            aggfunc="first"
        )
        csv_path = os.path.join(outdir, f"summary_auroc_{timestamp}.csv")
        pivot.to_csv(csv_path)
        print(f"Summary table → {csv_path}")
        print(f"\n{'='*60}")
        print("AUROC Summary Table")
        print(f"{'='*60}")
        print(pivot.to_string())

    return json_path


def main():
    ap = argparse.ArgumentParser(description="Run full evaluation pipeline")
    ap.add_argument("--model", required=True, help="HuggingFace model name/path")
    ap.add_argument("--datasets", nargs="+",
                    default=["injecagent", "advbench", "stealthy"],
                    choices=list(DATASET_REGISTRY.keys()),
                    help="Datasets to evaluate on")
    ap.add_argument("--baselines", nargs="+",
                    default=["linear", "mlp", "tfidf"],
                    choices=list(BASELINE_REGISTRY.keys()),
                    help="Baselines to run")
    ap.add_argument("--outdir", default="data/results",
                    help="Output directory for results")
    ap.add_argument("--load4bit", action="store_true")
    ap.add_argument("--llama_guard_model",
                    default="meta-llama/Llama-Guard-3-1B",
                    help="Llama Guard model to use when llama_guard baseline is selected")
    args = ap.parse_args()

    results = run_full_evaluation(
        model_name=args.model,
        datasets=args.datasets,
        baselines=args.baselines,
        outdir=args.outdir,
        load4bit=args.load4bit,
        llama_guard_model=args.llama_guard_model,
    )

    save_results(results, args.outdir)


if __name__ == "__main__":
    main()
