"""
Per-dataset breakdown of the rich sweep results.
Loads each model's cached spectral features, trains the probe on all data
(5-fold stratified CV), then evaluates ROC-AUC per source dataset using the
out-of-fold predictions — so every sample is scored exactly once by a probe
that never trained on it.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings("ignore")

DATA_ROOT = Path("c:/Users/valno/Dev/agent-safety/data/raw")
RESULTS_ROOT = Path("c:/Users/valno/Dev/agent-safety/data")

FAST_FILES = [
    ("advbench",   DATA_ROOT / "advbench_fast.jsonl"),
    ("stealthy",   DATA_ROOT / "stealthy_fast.jsonl"),
    ("agentdojo",  DATA_ROOT / "agentdojo_fast.jsonl"),
    ("injecagent", DATA_ROOT / "injecagent_fast.jsonl"),
    ("llmlat_benign", DATA_ROOT / "llmlat_benign_fast.jsonl"),
]

MODELS = [
    ("GPT-2",                  "rich_sweep_results"),
    ("Llama-3.2-3B-Instruct",  "rich_sweep_results_llama3b"),
    ("TinyLlama-1.1B-Chat",    "rich_sweep_results_tinyllama"),
    ("Llama-3.2-1B-Instruct",  "rich_sweep_results_llama1b"),
]

_SPECTRAL_METRICS = ["fiedler_value", "smoothness_index", "spectral_entropy", "energy", "hfer"]


def load_prompts_with_datasets():
    records = []
    for ds_name, fp in FAST_FILES:
        if not fp.exists():
            print(f"[warn] missing {fp}")
            continue
        for line in open(fp):
            rec = json.loads(line)
            records.append({
                "dataset": ds_name,
                "label": 1 if rec["label"] == "injected" else 0,
            })
    return records


def compute_rich_spectral_features(raw_samples: list) -> np.ndarray:
    N = len(raw_samples)
    ld0 = raw_samples[0].get("layer_diagnostics", [])
    L = len(ld0)
    M = len(_SPECTRAL_METRICS)
    T = np.zeros((N, L, M), dtype=np.float64)
    for i, s in enumerate(raw_samples):
        for li, layer in enumerate(s.get("layer_diagnostics", [])):
            for mi, m in enumerate(_SPECTRAL_METRICS):
                T[i, li, mi] = float(layer.get(m, 0.0) or 0.0)

    feats = []
    for mi in range(M):
        traj = T[:, :, mi]
        x = np.arange(L, dtype=float)
        q = max(1, L // 4)
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
        segs = np.array_split(np.arange(L), 4)
        for seg in segs:
            sl = traj[:, seg]
            feats.append(sl.mean(axis=1))
            feats.append(sl.std(axis=1))
        fft_mag = np.abs(np.fft.rfft(traj, axis=1)) / max(L, 1)
        for fi in range(fft_mag.shape[1]):
            feats.append(fft_mag[:, fi])

    X = np.column_stack(feats).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    meta = load_prompts_with_datasets()
    datasets = np.array([r["dataset"] for r in meta])
    y = np.array([r["label"] for r in meta])

    all_rows = []

    for model_name, cache_dir in MODELS:
        cache_path = RESULTS_ROOT / cache_dir / "spectral_cache.jsonl"
        if not cache_path.exists():
            print(f"[skip] {model_name} — no cache at {cache_path}")
            continue

        samples = [json.loads(l) for l in open(cache_path)]
        if len(samples) != len(meta):
            print(f"[skip] {model_name} — cache size mismatch ({len(samples)} vs {len(meta)})")
            continue

        X = compute_rich_spectral_features(samples)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

        for ds in np.unique(datasets):
            mask = datasets == ds
            y_ds = y[mask]
            s_ds = y_score[mask]
            if len(np.unique(y_ds)) < 2:
                auc = float("nan")
                pr_auc = float("nan")
                f1 = float("nan")
            else:
                auc = roc_auc_score(y_ds, s_ds)
                pr_auc = average_precision_score(y_ds, s_ds)
                thresh = 0.5
                f1 = f1_score(y_ds, (s_ds >= thresh).astype(int), zero_division=0)

            n_inj = y_ds.sum()
            n_ben = (1 - y_ds).sum()
            all_rows.append({
                "model": model_name,
                "dataset": ds,
                "n_injected": int(n_inj),
                "n_benign": int(n_ben),
                "roc_auc": round(float(auc), 4),
                "pr_auc": round(float(pr_auc), 4),
                "f1_at_0.5": round(float(f1), 4),
            })

        overall_auc = roc_auc_score(y, y_score)
        all_rows.append({
            "model": model_name,
            "dataset": "OVERALL",
            "n_injected": int(y.sum()),
            "n_benign": int((1 - y).sum()),
            "roc_auc": round(float(overall_auc), 4),
            "pr_auc": round(float(average_precision_score(y, y_score)), 4),
            "f1_at_0.5": round(float(f1_score(y, (y_score >= 0.5).astype(int), zero_division=0)), 4),
        })

    df = pd.DataFrame(all_rows)

    print("\n===== PER-DATASET ROC-AUC =====")
    pivot = df[df.dataset != "OVERALL"].pivot(index="dataset", columns="model", values="roc_auc")
    # order columns by overall AUC descending
    overall = df[df.dataset == "OVERALL"].set_index("model")["roc_auc"]
    col_order = overall.sort_values(ascending=False).index.tolist()
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string())

    print("\n===== OVERALL =====")
    print(df[df.dataset == "OVERALL"][["model", "roc_auc", "pr_auc"]].to_string(index=False))

    out_path = RESULTS_ROOT / "per_dataset_breakdown.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return df


if __name__ == "__main__":
    main()
