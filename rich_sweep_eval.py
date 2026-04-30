"""
Rich Sweep Evaluation of Agent-Safety data using spectral-trust metrics.

Pipeline:
  1. Load agent-safety JSONL prompts (fast subsets)
  2. Run spectral-trust GSPDiagnosticsFramework on each prompt -> layer_diagnostics
  3. Apply compute_rich_spectral_features (spectral-glaive rich sweep)
  4. Train logistic regression probe
  5. Multi-threshold × multi-metric × multi-layer evaluation table
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --spectral-trust imports --------------------------------------------------
from spectral_trust import (
    GSPConfig,
    GSPDiagnosticsFramework,
    SpectralAnalyzer,
    SpectralDiagnostics,
)

# --paths -------------------------------------------------------------------
DATA_ROOT = Path("c:/Users/valno/Dev/agent-safety/data/raw")
OUT_DIR   = Path("c:/Users/valno/Dev/agent-safety/data/rich_sweep_results_tinyllama")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_FILES = [
    DATA_ROOT / "advbench_fast.jsonl",
    DATA_ROOT / "stealthy_fast.jsonl",
    DATA_ROOT / "agentdojo_fast.jsonl",
    DATA_ROOT / "injecagent_fast.jsonl",
    DATA_ROOT / "llmlat_benign_fast.jsonl",
]

MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE      = "cuda"

_SPECTRAL_METRICS = ["fiedler_value", "smoothness_index", "spectral_entropy", "energy", "hfer"]

# --optimal threshold helper (from spectral-glaive) -------------------------
def find_optimal_threshold(y_true, y_score, target="f1"):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if target == "recall80":
        valid = np.where(recalls[:-1] >= 0.80)[0]
        return float(thresholds[valid[-1]]) if len(valid) else 0.5
    if target == "recall90":
        valid = np.where(recalls[:-1] >= 0.90)[0]
        return float(thresholds[valid[-1]]) if len(valid) else 0.5
    if target == "precision80":
        valid = np.where(precisions[:-1] >= 0.80)[0]
        return float(thresholds[valid[0]]) if len(valid) else 0.5
    # f1-optimal
    f1 = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1)])


# --rich sweep feature extraction (from spectral-glaive cli.py) -------------
def compute_rich_spectral_features(raw_samples: list) -> np.ndarray:
    N = len(raw_samples)
    if N == 0:
        return np.empty((0, 0), dtype=np.float32)
    ld0 = raw_samples[0].get("layer_diagnostics", [])
    L = len(ld0)
    if L == 0:
        return np.empty((N, 0), dtype=np.float32)

    M = len(_SPECTRAL_METRICS)
    T = np.zeros((N, L, M), dtype=np.float64)
    for i, s in enumerate(raw_samples):
        for li, layer in enumerate(s.get("layer_diagnostics", [])):
            for mi, m in enumerate(_SPECTRAL_METRICS):
                T[i, li, mi] = float(layer.get(m, 0.0) or 0.0)

    feats = []
    for mi in range(M):
        traj = T[:, :, mi]  # (N, L)
        x = np.arange(L, dtype=float)
        q = max(1, L // 4)

        # 15 trajectory statistics
        feats.append(traj.mean(axis=1))
        feats.append(traj.std(axis=1))
        feats.append(np.array([np.polyfit(x, t, 1)[0] for t in traj]))
        feats.append(traj[:, -q:].mean(axis=1) - traj[:, :q].mean(axis=1))

        diffs = np.diff(traj, axis=1) if L > 1 else np.zeros((N, 1))
        feats.append(diffs.min(axis=1))
        feats.append(diffs.max(axis=1))
        feats.append(diffs.std(axis=1))

        auc_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        feats.append(auc_fn(traj, axis=1) / max(L, 1))
        feats.append(traj.max(axis=1) - traj.min(axis=1))
        feats.append(sp_stats.skew(traj, axis=1))
        feats.append(sp_stats.kurtosis(traj, axis=1))

        m0, m1 = L // 3, max(L // 3 + 1, 2 * L // 3)
        feats.append(traj[:, m0:m1].mean(axis=1))

        early_m = traj[:, :q].mean(axis=1)
        late_m  = traj[:, -q:].mean(axis=1)
        feats.append(np.where(np.abs(early_m) > 1e-9, late_m / (early_m + 1e-9), 0.0))

        if L > 2:
            d2 = np.diff(traj, n=2, axis=1)
            feats.append((np.diff(np.sign(d2), axis=1) != 0).sum(axis=1).astype(float))
        else:
            feats.append(np.zeros(N))

        feats.append(traj.argmin(axis=1).astype(float) / max(L - 1, 1))

        # 8 segmental features (4 segments × mean+std)
        segs = np.array_split(np.arange(L), 4)
        for seg in segs:
            sl = traj[:, seg]
            feats.append(sl.mean(axis=1))
            feats.append(sl.std(axis=1))

        # FFT magnitude (half-spectrum, normalised)
        fft_mag = np.abs(np.fft.rfft(traj, axis=1)) / max(L, 1)
        for fi in range(fft_mag.shape[1]):
            feats.append(fft_mag[:, fi])

    X = np.column_stack(feats).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# --per-layer single-metric feature (for layer × metric sweep) --------------
def single_layer_metric_features(raw_samples: list, layer_idx: int, metric: str) -> np.ndarray:
    out = []
    for s in raw_samples:
        ld = s.get("layer_diagnostics", [])
        if layer_idx < len(ld):
            val = float(ld[layer_idx].get(metric, 0.0) or 0.0)
        else:
            val = 0.0
        out.append(val)
    return np.array(out, dtype=np.float32).reshape(-1, 1)


# --load data ----------------------------------------------------------------
def load_prompts():
    prompts, labels = [], []
    for fp in FAST_FILES:
        if not fp.exists():
            print(f"[warn] missing {fp}")
            continue
        for line in open(fp):
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            labels.append(1 if rec["label"] == "injected" else 0)
    print(f"Loaded {len(prompts)} samples  ({sum(labels)} injected, {len(labels)-sum(labels)} benign)")
    return prompts, labels


# --run spectral-trust -------------------------------------------------------
def run_spectral_analysis(prompts: list) -> list:
    """
    Uses spectral_trust.GSPDiagnosticsFramework to compute per-layer spectral
    metrics for each prompt.  Returns a list of dicts each containing
    'layer_diagnostics' (list of metric dicts, one per layer).
    """
    cache_path = OUT_DIR / "spectral_cache.jsonl"
    if cache_path.exists():
        cached = [json.loads(l) for l in open(cache_path)]
        if len(cached) == len(prompts):
            print(f"Loaded {len(cached)} cached spectral results.")
            return cached

    config = GSPConfig(
        model_name=MODEL_NAME,
        device=DEVICE,
        output_dir=str(OUT_DIR / "gsp_framework"),
        verbose=False,
        save_plots=False,
        save_intermediate=False,
    )
    framework = GSPDiagnosticsFramework(config)
    framework.instrumenter.load_model(MODEL_NAME)

    samples = []
    with open(cache_path, "w") as fout:
        for prompt in tqdm(prompts, desc="spectral-trust analysis"):
            result = framework.analyze_text(prompt, save_results=False)
            ld_raw = result.get("layer_diagnostics", [])
            # Normalise each layer dict to only the 5 spectral metrics
            ld = []
            for layer_obj in ld_raw:
                if isinstance(layer_obj, dict):
                    ld.append({m: float(layer_obj.get(m, 0.0) or 0.0) for m in _SPECTRAL_METRICS})
                else:
                    # SpectralDiagnostics dataclass
                    ld.append({m: float(getattr(layer_obj, m, 0.0) or 0.0) for m in _SPECTRAL_METRICS})
            rec = {"layer_diagnostics": ld}
            samples.append(rec)
            fout.write(json.dumps(rec) + "\n")

    print(f"Spectral analysis done. {len(samples)} samples, {len(samples[0]['layer_diagnostics'])} layers each.")
    return samples


# --evaluation helpers --------------------------------------------------------
def evaluate_probe(X: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    roc_auc = roc_auc_score(y, y_score)
    pr_auc  = average_precision_score(y, y_score)

    thresholds = {
        "t_f1":        find_optimal_threshold(y, y_score, "f1"),
        "t_recall80":  find_optimal_threshold(y, y_score, "recall80"),
        "t_recall90":  find_optimal_threshold(y, y_score, "recall90"),
        "t_prec80":    find_optimal_threshold(y, y_score, "precision80"),
    }

    rows = []
    for tname, thresh in thresholds.items():
        y_pred = (y_score >= thresh).astype(int)
        rows.append({
            "threshold_target": tname,
            "threshold_value":  round(thresh, 4),
            "f1":       round(f1_score(y, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall":   round(recall_score(y, y_pred, zero_division=0), 4),
            "accuracy": round(accuracy_score(y, y_pred), 4),
        })

    return {
        "label":   label,
        "roc_auc": round(roc_auc, 4),
        "pr_auc":  round(pr_auc, 4),
        "thresholds": rows,
        "y_score": y_score,
    }


# --per-layer × per-metric sweep ---------------------------------------------
def layer_metric_sweep(samples: list, y: np.ndarray, n_layers: int) -> pd.DataFrame:
    rows = []
    for layer_idx in range(n_layers):
        for metric in _SPECTRAL_METRICS:
            X = single_layer_metric_features(samples, layer_idx, metric)
            if X.std() < 1e-8:
                continue
            clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
                auc = roc_auc_score(y, y_score)
            except Exception:
                auc = 0.5
            rows.append({"layer": layer_idx, "metric": metric, "roc_auc": round(auc, 4)})
    return pd.DataFrame(rows)


# --main ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  AGENT-SAFETY  ×  SPECTRAL-TRUST  RICH SWEEP EVALUATION")
    print("=" * 70)

    # 1. Load agent-safety prompts
    prompts, labels = load_prompts()
    y = np.array(labels)

    # 2. Spectral analysis via spectral-trust
    print(f"\n[1] Running spectral-trust ({MODEL_NAME}) on {len(prompts)} prompts ...")
    samples = run_spectral_analysis(prompts)

    n_layers = len(samples[0]["layer_diagnostics"])
    print(f"    Layers per sample: {n_layers}")

    # 3. Rich sweep feature matrix
    print("\n[2] Computing rich spectral feature matrix ...")
    X_rich = compute_rich_spectral_features(samples)
    print(f"    Feature shape: {X_rich.shape}  (N × D)")

    # 4. Evaluate rich probe
    print("\n[3] Cross-validating rich spectral probe (5-fold stratified) ...")
    rich_result = evaluate_probe(X_rich, y, label="rich_sweep")

    print(f"\n    ROC-AUC : {rich_result['roc_auc']:.4f}")
    print(f"    PR-AUC  : {rich_result['pr_auc']:.4f}")

    # 5. Multi-threshold table
    print("\n--Multi-Threshold Performance --------------------------------------")
    tdf = pd.DataFrame(rich_result["thresholds"])
    print(tdf.to_string(index=False))

    # 6. Per-layer × per-metric sweep
    print(f"\n[4] Per-layer × per-metric sweep ({n_layers} layers × {len(_SPECTRAL_METRICS)} metrics) ...")
    lm_df = layer_metric_sweep(samples, y, n_layers)

    # Best per metric
    print("\n--Best Layer per Metric (ROC-AUC) ----------------------------------")
    best_per_metric = lm_df.groupby("metric")["roc_auc"].agg(["max", "idxmax"])
    for metric, row in best_per_metric.iterrows():
        best_layer = lm_df.loc[int(row["idxmax"]), "layer"]
        print(f"  {metric:<22}  best_layer={best_layer:>2}  AUC={row['max']:.4f}")

    # Best overall
    top10 = lm_df.nlargest(10, "roc_auc")[["layer", "metric", "roc_auc"]]
    print("\n--Top-10 (layer, metric) Pairs -------------------------------------")
    print(top10.to_string(index=False))

    # Heatmap-ready pivot
    pivot = lm_df.pivot(index="layer", columns="metric", values="roc_auc")
    pivot_path = OUT_DIR / "layer_metric_auc_heatmap.csv"
    pivot.to_csv(pivot_path)
    print(f"\n    Pivot table saved to {pivot_path}")

    # 7. Per-metric trajectory probe comparison
    print("\n[5] Per-metric trajectory probe (all layers combined, 1 metric at a time) ...")
    metric_results = []
    for metric in _SPECTRAL_METRICS:
        T = np.array([[ld.get(metric, 0.0) for ld in s["layer_diagnostics"]] for s in samples], dtype=np.float32)
        T = np.nan_to_num(T)
        res = evaluate_probe(T, y, label=metric)
        metric_results.append({"metric": metric, "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]})

    print("\n--Per-Metric Trajectory Probe -------------------------------------")
    mdf = pd.DataFrame(metric_results).sort_values("roc_auc", ascending=False)
    print(mdf.to_string(index=False))

    # 8. Summary comparison
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model             : {MODEL_NAME}  ({n_layers} layers)")
    print(f"  Dataset           : agent-safety fast splits")
    print(f"  N samples         : {len(prompts)}  ({y.sum()} injected / {(1-y).sum()} benign)")
    print(f"  Feature dim (rich): {X_rich.shape[1]}")
    print(f"")
    print(f"  Rich sweep ROC-AUC: {rich_result['roc_auc']:.4f}")
    print(f"  Rich sweep PR-AUC : {rich_result['pr_auc']:.4f}")
    best_single = mdf.iloc[0]
    print(f"  Best single metric: {best_single['metric']}  AUC={best_single['roc_auc']:.4f}")
    best_lm = lm_df.nlargest(1, "roc_auc").iloc[0]
    print(f"  Best (layer,metric): L{int(best_lm.layer)}_{best_lm.metric}  AUC={best_lm.roc_auc:.4f}")

    # Save results
    results_path = OUT_DIR / "rich_sweep_summary.json"
    summary = {
        "model": MODEL_NAME,
        "n_samples": len(prompts),
        "n_injected": int(y.sum()),
        "n_benign": int((1 - y).sum()),
        "n_layers": n_layers,
        "feature_dim_rich": int(X_rich.shape[1]),
        "rich_roc_auc": rich_result["roc_auc"],
        "rich_pr_auc": rich_result["pr_auc"],
        "multi_threshold": rich_result["thresholds"],
        "per_metric": metric_results,
        "top10_layer_metric": top10.to_dict(orient="records"),
    }
    json.dump(summary, open(results_path, "w"), indent=2)
    print(f"\n  Full results saved to {results_path}")

    return summary


if __name__ == "__main__":
    main()
