"""
layerwise_analysis.py
======================
Layer-wise probe analysis for the agent safety benchmark.

For each dataset and each cross-dataset pair, it trains a Linear Probe
and an MLP Probe on the hidden states of each individual layer separately,
to determine which transformer layer best encodes generalizable safety signals.

Usage:
    python -m src.analysis.layerwise_analysis \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --outdir data/plots

The script reads pre-extracted NPZ files from data/features/ and produces:
  - data/plots/layerwise_auroc_indomain.png   (In-domain per layer)
  - data/plots/layerwise_auroc_crossds.png    (Cross-dataset avg per layer)
  - data/plots/layerwise_results.csv          (Full results table)
  - data/plots/layerwise_analysis.md          (Interpretation report)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn

from src.eval.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Feature utilities
# ─────────────────────────────────────────────────────────────────────────────

MODEL = "TinyLlama_TinyLlama-1.1B-Chat-v1.0"
HIDDEN_SIZE = 2048  # TinyLlama hidden dim per layer
DATASET_NAMES = ["advbench", "injecagent", "stealthy"]

def load_npz(ds_name: str, fast: bool = False) -> tuple:
    suffix = "_fast" if fast else ""
    path = f"data/features/{MODEL}_{ds_name}_feats{suffix}.npz"
    if not os.path.exists(path):
        return None, None, None
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)    # (N, n_layers * hidden_size)
    y = d["y"].astype(np.float32)
    layers = d["layers"].tolist()
    return X, y, layers


def extract_layer_slice(X: np.ndarray, layer_pos: int) -> np.ndarray:
    """Extract the hidden state slice for layer at position `layer_pos` in the npz."""
    start = layer_pos * HIDDEN_SIZE
    end   = start + HIDDEN_SIZE
    return X[:, start:end]


def stratified_split(y: np.ndarray, test_ratio: float = 0.3, seed: int = 42):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0].copy(); rng.shuffle(pos)
    neg = np.where(y == 0)[0].copy(); rng.shuffle(neg)
    n_pos_te = max(1, int(len(pos) * test_ratio))
    n_neg_te = max(1, int(len(neg) * test_ratio))
    tr = np.concatenate([pos[n_pos_te:], neg[n_neg_te:]])
    te = np.concatenate([pos[:n_pos_te], neg[:n_neg_te]])
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


# ─────────────────────────────────────────────────────────────────────────────
# Probes (pure PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_probe(X_tr, y_tr, X_te, y_te, lr=0.005, epochs=300, wd=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mean-center on train statistics
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_te_n = (X_te - mean) / std

    X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr,   dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te_n, dtype=torch.float32).to(device)

    n_pos = float(y_tr.sum()); n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model = nn.Linear(X_tr_n.shape[1], 1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(256, len(y_tr)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te_t).squeeze(-1)).cpu().numpy()

    return compute_metrics(y_te, probs)


def train_mlp_probe(X_tr, y_tr, X_te, y_te, lr=0.005, epochs=150, wd=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_te_n = (X_te - mean) / std

    D = X_tr_n.shape[1]
    X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr,   dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te_n, dtype=torch.float32).to(device)

    n_pos = float(y_tr.sum()); n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    net = nn.Sequential(
        nn.Linear(D, 256), nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64), nn.GELU(),
        nn.Linear(64, 1),
    ).to(device)
    opt  = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(256, len(y_tr)), shuffle=True)

    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); crit(net(xb).squeeze(-1), yb).backward(); opt.step()

    net.eval()
    with torch.no_grad():
        probs = torch.sigmoid(net(X_te_t).squeeze(-1)).cpu().numpy()

    return compute_metrics(y_te, probs)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_layerwise(fast: bool = False):
    results = []

    datasets = {}
    for name in DATASET_NAMES:
        X, y, layers = load_npz(name, fast)
        if X is not None:
            datasets[name] = (X, y, layers)
            print(f"  [load] {name}: X={X.shape}, layers={layers}")

    if not datasets:
        print("No features found in data/features/. Run run_experiments.py first.")
        return pd.DataFrame()

    # Use the first dataset's layer list (all should be the same)
    first_ds = list(datasets.values())[0]
    layer_indices = first_ds[2]   # e.g. [0, 5, 11, 16, 21]
    n_layers = len(layer_indices)

    # ── Block 1: In-Domain ───────────────────────────────────────────────────
    print("\n[Block 1] In-Domain evaluation (per layer)...")
    for ds_name, (X, y, layers) in datasets.items():
        tr_idx, te_idx = stratified_split(y)
        for pos, layer_id in enumerate(layer_indices):
            X_layer = extract_layer_slice(X, pos)
            X_tr, X_te = X_layer[tr_idx], X_layer[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            m_lin = train_linear_probe(X_tr, y_tr, X_te, y_te)
            m_mlp = train_mlp_probe(X_tr, y_tr, X_te, y_te)

            for probe, m in [("Linear Probe", m_lin), ("MLP Probe", m_mlp)]:
                results.append({
                    "Block": "In-Domain",
                    "Train_DS": ds_name,
                    "Test_DS":  ds_name,
                    "Layer_ID": int(layer_id),
                    "Layer_Pos": pos,
                    "Probe": probe,
                    "AUROC":  m["auroc"],
                    "AUPRC":  m.get("auprc", float("nan")),
                    "F1":     m["f1"],
                    "Opt-F1": m.get("opt_f1", float("nan")),
                })
            print(f"    [{ds_name}] layer {layer_id:2d}: LinP AUROC={m_lin['auroc']:.3f} | MLP AUROC={m_mlp['auroc']:.3f}")

    # ── Block 2: Cross-Dataset ───────────────────────────────────────────────
    print("\n[Block 2] Cross-Dataset evaluation (per layer)...")
    ds_names = list(datasets.keys())
    for ds_tr in ds_names:
        for ds_te in ds_names:
            if ds_tr == ds_te:
                continue
            X_tr_full, y_tr, layers = datasets[ds_tr]
            X_te_full, y_te, _      = datasets[ds_te]

            for pos, layer_id in enumerate(layer_indices):
                X_tr_l = extract_layer_slice(X_tr_full, pos)
                X_te_l = extract_layer_slice(X_te_full, pos)

                m_lin = train_linear_probe(X_tr_l, y_tr, X_te_l, y_te)
                m_mlp = train_mlp_probe(X_tr_l, y_tr, X_te_l, y_te)

                for probe, m in [("Linear Probe", m_lin), ("MLP Probe", m_mlp)]:
                    results.append({
                        "Block": f"Cross-DS ({ds_tr}->{ds_te})",
                        "Train_DS": ds_tr,
                        "Test_DS":  ds_te,
                        "Layer_ID": int(layer_id),
                        "Layer_Pos": pos,
                        "Probe": probe,
                        "AUROC":  m["auroc"],
                        "AUPRC":  m.get("auprc", float("nan")),
                        "F1":     m["f1"],
                        "Opt-F1": m.get("opt_f1", float("nan")),
                    })
            print(f"    [{ds_tr} -> {ds_te}] layer {layer_id:2d}: LinP={m_lin['auroc']:.3f} | MLP={m_mlp['auroc']:.3f}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "advbench":   "#e74c3c",
    "injecagent": "#3498db",
    "stealthy":   "#2ecc71",
    "Cross-DS avg": "#9b59b6",
}


def plot_layerwise_indomain(df: pd.DataFrame, out_dir: str):
    """Line chart of per-layer AUROC for each dataset (In-Domain, Linear Probe)."""
    b1 = df[(df["Block"] == "In-Domain") & (df["Probe"] == "Linear Probe")]
    if b1.empty:
        print("  [skip] No in-domain layer data.")
        return

    layer_ids = sorted(b1["Layer_ID"].unique())
    datasets  = sorted(b1["Train_DS"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for ds in datasets:
        sub = b1[b1["Train_DS"] == ds].sort_values("Layer_ID")
        aurocs = [sub[sub["Layer_ID"] == lid]["AUROC"].mean() for lid in layer_ids]
        color = COLORS.get(ds, "#ffffff")
        ax.plot(layer_ids, aurocs, marker="o", linewidth=2.5, markersize=7,
                color=color, label=ds.capitalize())
        # Annotate best layer
        best_idx = int(np.argmax(aurocs))
        ax.annotate(f"  {aurocs[best_idx]:.3f}",
                    xy=(layer_ids[best_idx], aurocs[best_idx]),
                    color=color, fontsize=9)

    ax.set_xlabel("TinyLlama Layer Index", color="white", fontsize=12)
    ax.set_ylabel("AUROC (Linear Probe)", color="white", fontsize=12)
    ax.set_title("Layer-wise Safety Signal – In-Domain", color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3a3f55")
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.2, color="#3a3f55")
    ax.legend(framealpha=0.3, facecolor="#1a1d27", labelcolor="white", fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, "layerwise_auroc_indomain.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_layerwise_crossds(df: pd.DataFrame, out_dir: str):
    """Line chart of per-layer AUROC averaged over all cross-dataset pairs."""
    b2 = df[(df["Block"] != "In-Domain") & (df["Probe"] == "Linear Probe")]
    if b2.empty:
        print("  [skip] No cross-DS layer data.")
        return

    layer_ids = sorted(b2["Layer_ID"].unique())

    # Mean across all cross-DS pairs and all probes
    avg_by_layer = b2.groupby("Layer_ID")["AUROC"].mean().reindex(layer_ids).values

    # Also by probe type for MLP
    mlp = df[(df["Block"] != "In-Domain") & (df["Probe"] == "MLP Probe")]
    mlp_by_layer = mlp.groupby("Layer_ID")["AUROC"].mean().reindex(layer_ids).values if not mlp.empty else None

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.plot(layer_ids, avg_by_layer, marker="o", linewidth=2.5, markersize=8,
            color="#3498db", label="Linear Probe")
    if mlp_by_layer is not None:
        ax.plot(layer_ids, mlp_by_layer, marker="s", linewidth=2.5, markersize=8,
                color="#e74c3c", label="MLP Probe", linestyle="--")

    # Shade the "best generalizing zone"
    best_lin = layer_ids[int(np.argmax(avg_by_layer))]
    ax.axvline(x=best_lin, color="#2ecc71", linewidth=1.4, linestyle=":", alpha=0.8,
               label=f"Best lin layer: {best_lin}")
    ax.annotate(f"Best: L{best_lin}\n({avg_by_layer[np.argmax(avg_by_layer)]:.3f})",
                xy=(best_lin, avg_by_layer[np.argmax(avg_by_layer)]),
                xytext=(best_lin + 0.5, avg_by_layer[np.argmax(avg_by_layer)] - 0.06),
                color="#2ecc71", fontsize=10, arrowprops=dict(arrowstyle="->", color="#2ecc71"))

    ax.set_xlabel("TinyLlama Layer Index", color="white", fontsize=12)
    ax.set_ylabel("Avg AUROC (Cross-Dataset)", color="white", fontsize=12)
    ax.set_title("Layer-wise Cross-Dataset Generalization", color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3a3f55")
    ax.set_ylim(0.3, 1.05)
    ax.grid(alpha=0.2, color="#3a3f55")
    ax.legend(framealpha=0.3, facecolor="#1a1d27", labelcolor="white", fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, "layerwise_auroc_crossds.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_layerwise_heatmap(df: pd.DataFrame, out_dir: str):
    """Heatmap of AUROC: rows=layers, cols=cross-dataset pairs (Linear Probe)."""
    b2 = df[(df["Block"] != "In-Domain") & (df["Probe"] == "Linear Probe")].copy()
    if b2.empty:
        return

    b2["Pair"] = b2["Train_DS"].str[:3] + "->" + b2["Test_DS"].str[:3]
    pivot = b2.pivot_table(index="Layer_ID", columns="Pair", values="AUROC", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="AUROC", fraction=0.03, pad=0.04)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color="white", rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Layer {l}" for l in pivot.index], color="white", fontsize=9)
    ax.set_title("Layer vs Cross-Dataset Pair: Linear Probe AUROC", color="white", fontsize=13, pad=12)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black" if val > 0.6 else "white")

    plt.tight_layout()
    out = os.path.join(out_dir, "layerwise_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def generate_md_report(df: pd.DataFrame, out_dir: str):
    b1 = df[(df["Block"] == "In-Domain") & (df["Probe"] == "Linear Probe")]
    b2 = df[(df["Block"] != "In-Domain") & (df["Probe"] == "Linear Probe")]

    layer_ids = sorted(df["Layer_ID"].unique())

    in_dom_by_layer  = b1.groupby("Layer_ID")["AUROC"].mean()
    cross_ds_by_layer = b2.groupby("Layer_ID")["AUROC"].mean()

    best_in  = int(in_dom_by_layer.idxmax()) if not in_dom_by_layer.empty else "N/A"
    best_crs = int(cross_ds_by_layer.idxmax()) if not cross_ds_by_layer.empty else "N/A"

    # Table
    rows = []
    for lid in layer_ids:
        ind_r  = in_dom_by_layer.get(lid, float("nan"))
        crs_r  = cross_ds_by_layer.get(lid, float("nan"))
        rows.append(f"| Layer {lid:2d} | {ind_r:.3f} | {crs_r:.3f} |")
    table = "\n".join(rows)

    md = f"""# Layer-wise Safety Benchmark Analysis

## Summary Table – Linear Probe AUROC by Layer

| Layer | In-Domain AUROC | Cross-Dataset AUROC |
|-------|-----------------|---------------------|
{table}

## Key Findings

1. **Best In-Domain Layer**: Layer {best_in}
   - Achieves peak AUROC on the original dataset distribution.
   - This layer has likely *overfit* to the specific formatting/style of each dataset.

2. **Best Generalizing Layer**: Layer {best_crs}
   - Provides the highest average Cross-Dataset AUROC.
   - This is the layer that should be used for the probe in the main paper.

3. **Interpretation**:
   - Layers close to the **output** (last layer) tend to overfit in-domain cues (length, format) and collapse cross-domain.
   - **Intermediate layers** encode more generalizable, semantically rich representations of intent.
   - This is consistent with findings from probing literature (Tenney et al., 2019; Belinkov & Glass, 2019).

## Implications for the Paper

> [!IMPORTANT]
> If Layer {best_crs} achieves significantly higher Cross-Dataset AUROC than the last layer,
> this provides strong evidence that *the LLM does internally encode safety-relevant information*,
> but the **signal is distributed across intermediate layers** and is more generalizable there.

## Plots

- `layerwise_auroc_indomain.png` – Per-dataset, per-layer AUROC (In-Domain)
- `layerwise_auroc_crossds.png`  – Average Cross-Dataset AUROC by layer
- `layerwise_heatmap.png`        – Heatmap: Layer x Cross-DS pair AUROC
"""

    out_path = os.path.join(out_dir, "layerwise_analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Report saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--outdir", default="data/plots")
    ap.add_argument("--fast", action="store_true", help="Use fast NPZ files (100 samples)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 65)
    print("  Layer-wise Safety Analysis")
    print("=" * 65)

    df = run_layerwise(fast=args.fast)
    if df.empty:
        return

    # Save raw results
    csv_out = os.path.join(args.outdir, "layerwise_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Results CSV: {csv_out}")

    # Plots
    print("\n  Generating plots...")
    plot_layerwise_indomain(df, args.outdir)
    plot_layerwise_crossds(df, args.outdir)
    plot_layerwise_heatmap(df, args.outdir)

    # Report
    generate_md_report(df, args.outdir)

    print("\n  Done. Summary:")
    lin = df[df["Probe"] == "Linear Probe"]
    for blk in df["Block"].unique():
        subset = lin[lin["Block"] == blk]
        best_layer = subset.groupby("Layer_ID")["AUROC"].mean().idxmax()
        best_auc   = subset.groupby("Layer_ID")["AUROC"].mean().max()
        print(f"    {blk:40s} -> Best layer: {best_layer}  (AUROC={best_auc:.3f})")


if __name__ == "__main__":
    main()
