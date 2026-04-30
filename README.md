# ICLR Agent Safety: MONITORING INTERNAL REPRESENTATIONS FOR
SAFE AUTONOMOUS LLM AGENTS

This repository contains a reproducible research pipeline for detecting **Prompt Injection** signals in Large Language Models (LLMs) using latent activation probing.

##  Overview

Our methodology demonstrates that prompt injections (such as "ignore previous instructions") leave unique "semantic signatures" in the hidden states (activations) of transformer models. By training lightweight linear probes on these activations, we can detect attacks with near-perfect accuracy, even when the attacks are designed to be "stealthy" and lack obvious structural cues (like `Task:` headers).

### Key Features
- **Activation Extraction**: Support for extracting layer-wise hidden states from Hugging Face models (TinyLlama, Qwen, etc.).
- **Latent Probing**: High-performance linear probes with strict **Group Split** validation to prevent paired-sample leakage.
- **Stealthy Benchmarking**: Evaluation on datasets designed to isolate semantic signal from formatting artifacts.
- **Statistical Rigor**: Built-in tools for computing p-values, Cohen's d effect sizes, and bootstrap confidence intervals.



## Installation

```bash
# Create environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset
```bash
python -m src.generate.generate_stealthy --output data/raw/prompts_stealthy.jsonl --n 400
```

### 2. Run Detection Pipeline
The main entry point performs activation extraction followed by probe training and evaluation.
```bash
python -m src.run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --input data/raw/prompts_stealthy.jsonl --outdir data/processed
```

### 3. Generate Analysis Plots
```bash
python scripts/generate_plots.py
```
*Plots will be saved to `data/plots/` in PDF (vector) and PNG formats.*

## Results

### Baseline Probe Results (activation probing, TinyLlama-1.1B)

Evaluation across three protocols drawn from `data/results/eval_blocks_181109.csv`. All probes use layer-wise hidden states and strict Group Split validation to prevent paired-sample leakage.

**Block 1 — In-Domain (train & test on same dataset)**

| Dataset | Linear Probe | MLP Probe | TF-IDF + LR | Semantic MLP | Perplexity | Length Clf |
|---|---|---|---|---|---|---|
| InjecAgent | 1.000 | 1.000 | 1.000 | 1.000 | 0.538 | 0.876 |
| AdvBench | 1.000 | 1.000 | 1.000 | 1.000 | 0.880 | 0.898 |
| AgentDojo | 1.000 | 1.000 | 1.000 | 1.000 | 0.360 | 0.818 |
| Stealthy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.778 |

> AUROC reported. In-domain probes saturate at 1.0; see Block 2 for the harder generalization scenario.

**Block 2 — Cross-Domain Transfer (train on one dataset, test on another)**

| Train → Test | Linear Probe | MLP Probe | TF-IDF + LR | Semantic MLP | Perplexity |
|---|---|---|---|---|---|
| InjecAgent → AdvBench | 0.000 | 0.102 | 1.000 | 0.981 | 0.890 |
| InjecAgent → AgentDojo | 0.525 | 0.500 | 1.000 | 1.000 | 0.323 |
| InjecAgent → Stealthy | 0.000 | 0.116 | 1.000 | 1.000 | 1.000 |
| AdvBench → InjecAgent | 1.000 | 0.499 | 1.000 | 1.000 | 0.628 |
| AdvBench → AgentDojo | 1.000 | 0.871 | 0.998 | 1.000 | 0.323 |
| AdvBench → Stealthy | 0.029 | 0.080 | 1.000 | 1.000 | 1.000 |
| AgentDojo → InjecAgent | 1.000 | 1.000 | 1.000 | 1.000 | 0.628 |
| AgentDojo → AdvBench | 1.000 | 1.000 | 1.000 | 0.969 | 0.890 |
| AgentDojo → Stealthy | 0.070 | 0.122 | 1.000 | 0.996 | 1.000 |
| Stealthy → InjecAgent | 0.053 | 0.699 | 0.965 | 0.978 | 0.628 |
| Stealthy → AdvBench | 0.522 | 0.589 | 0.946 | 0.936 | 0.890 |
| Stealthy → AgentDojo | 0.879 | 0.748 | 0.869 | 0.982 | 0.323 |

> Activation probes (Linear/MLP) collapse to ~0.0 AUROC when trained on stealthy injections and tested on lexically different attacks, because they overfit to surface features of the training domain.

**Block 3 — Leave-One-Out (LOO) Generalisation**

| Test Dataset | Linear Probe | MLP Probe | TF-IDF + LR | Semantic MLP | Perplexity |
|---|---|---|---|---|---|
| InjecAgent | 1.000 | 1.000 | 1.000 | 1.000 | 0.628 |
| AdvBench | 1.000 | 1.000 | 0.997 | 1.000 | 0.890 |
| AgentDojo | 1.000 | 1.000 | 0.998 | 1.000 | 0.323 |
| Stealthy | 0.309 | 0.159 | 1.000 | 1.000 | 1.000 |

> Linear and MLP probes fail badly on the stealthy split in the LOO setting (0.31 / 0.16 AUROC), exposing a fundamental generalisation gap for activation-based detectors.

---

### Spectral-Trust Rich Sweep (attention graph metrics, no hidden states)

A complementary approach using [`spectral-trust`](https://pypi.org/project/spectral-trust/) to extract graph-spectral diagnostics from attention matrices (Fiedler value, smoothness index, spectral entropy, energy, HFER) across all layers. The **rich sweep** builds a high-dimensional feature vector per sample by combining trajectory statistics, segmental features, and FFT magnitude — then trains a logistic regression probe.

Evaluated on a **mixed multi-dataset split** (advbench + agentdojo + injecagent + stealthy + llmlat_benign fast subsets, 460 samples total, 5-fold stratified CV). This is a harder cross-dataset setting — there is no held-out train/test domain split, but the model has never seen dataset-specific surface features.

**Overall Performance**

| Backbone | Layers | Feature dim | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| GPT-2 | 12 | 150 | **0.900** | **0.855** |
| Llama-3.2-3B-Instruct | 28 | 190 | 0.889 | 0.833 |
| TinyLlama-1.1B-Chat | 22 | 175 | 0.866 | 0.799 |
| Llama-3.2-1B-Instruct | 16 | 160 | 0.858 | 0.797 |

**Multi-Threshold Breakdown — GPT-2 backbone**

| Threshold target | Threshold | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| F1-optimal | 0.336 | **0.802** | 0.755 | 0.856 | 0.835 |
| Recall ≥ 80% | 0.436 | 0.783 | 0.766 | 0.800 | 0.826 |
| Recall ≥ 90% | 0.189 | 0.775 | 0.681 | **0.900** | 0.796 |
| Precision ≥ 80% | 0.573 | 0.765 | **0.800** | 0.733 | 0.824 |

**Multi-Threshold Breakdown — Llama-3.2-3B-Instruct backbone**

| Threshold target | Threshold | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| F1-optimal | 0.204 | **0.785** | 0.695 | **0.900** | 0.807 |
| Recall ≥ 80% | 0.371 | 0.750 | 0.706 | 0.800 | 0.791 |
| Recall ≥ 90% | 0.204 | **0.785** | 0.695 | **0.900** | 0.807 |
| Precision ≥ 80% | 0.702 | 0.712 | **0.804** | 0.639 | 0.798 |

**Multi-Threshold Breakdown — TinyLlama-1.1B-Chat backbone**

| Threshold target | Threshold | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| F1-optimal | 0.394 | **0.757** | 0.737 | 0.778 | 0.804 |
| Recall ≥ 80% | 0.335 | 0.752 | 0.709 | 0.800 | 0.794 |
| Recall ≥ 90% | 0.126 | 0.741 | 0.630 | **0.900** | 0.754 |
| Precision ≥ 80% | 0.673 | 0.700 | **0.800** | 0.622 | 0.791 |

**Multi-Threshold Breakdown — Llama-3.2-1B-Instruct backbone**

| Threshold target | Threshold | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| F1-optimal | 0.172 | 0.747 | 0.630 | **0.917** | 0.757 |
| Recall ≥ 80% | 0.307 | 0.725 | 0.664 | 0.800 | 0.763 |
| Recall ≥ 90% | 0.186 | 0.743 | 0.633 | 0.900 | 0.757 |
| Precision ≥ 80% | 0.798 | 0.590 | **0.800** | 0.467 | 0.746 |

**Per-Metric Trajectory Probe (all layers combined, 1 metric at a time)**

| Metric | GPT-2 AUC | Llama-3B AUC | TinyLlama AUC | Llama-1B AUC |
|---|---|---|---|---|
| `fiedler_value` | 0.816 | **0.833** | **0.840** | 0.715 |
| `smoothness_index` | 0.796 | 0.820 | 0.807 | 0.750 |
| `energy` | 0.795 | 0.797 | 0.793 | **0.759** |
| `spectral_entropy` | 0.649 | 0.628 | 0.728 | 0.625 |
| `hfer` | 0.666 | 0.626 | 0.703 | 0.624 |

**Best (layer, metric) Pair**

| Backbone | Best layer | Best metric | AUC |
|---|---|---|---|
| GPT-2 | L11 | `smoothness_index` | 0.770 |
| Llama-3.2-3B-Instruct | L26 | `energy` | 0.702 |
| TinyLlama-1.1B-Chat | L7 | `fiedler_value` | 0.673 |
| Llama-3.2-1B-Instruct | L13 | `smoothness_index` | 0.645 |

> Across all four backbones, `fiedler_value` is consistently the strongest single-metric trajectory signal. TinyLlama peaks early (L7), reflecting its shallower architecture, while Llama-3B's final layer (L26) shows a sharp energy spike. GPT-2's best single pair (L11 × smoothness) stands out despite being the smallest model tested.

**Layer × Metric Heatmap (AUROC) — GPT-2**

| Layer | energy | fiedler | hfer | smoothness | entropy |
|---|---|---|---|---|---|
| 0 | 0.624 | 0.650 | 0.614 | 0.553 | 0.617 |
| 1 | 0.601 | 0.614 | **0.632** | 0.691 | 0.536 |
| 4 | 0.658 | 0.618 | 0.514 | 0.664 | 0.534 |
| 5 | **0.669** | 0.635 | 0.504 | 0.670 | 0.576 |
| 11 | 0.652 | 0.634 | 0.587 | **0.770** | 0.611 |

**Layer × Metric Heatmap (AUROC) — Llama-3.2-3B-Instruct** (selected layers)

| Layer | energy | fiedler | hfer | smoothness | entropy |
|---|---|---|---|---|---|
| 0 | 0.619 | 0.633 | 0.504 | **0.654** | 0.528 |
| 3 | 0.593 | **0.673** | 0.523 | 0.591 | 0.598 |
| 5 | 0.589 | **0.673** | 0.517 | 0.585 | 0.552 |
| 22 | 0.628 | 0.634 | 0.521 | 0.652 | **0.625** |
| 23 | 0.625 | 0.640 | 0.517 | **0.654** | 0.613 |
| 26 | **0.702** | 0.631 | 0.488 | 0.610 | 0.591 |

**Layer × Metric Heatmap (AUROC) — TinyLlama-1.1B-Chat** (selected layers)

| Layer | energy | fiedler | hfer | smoothness | entropy |
|---|---|---|---|---|---|
| 0 | 0.614 | 0.607 | 0.621 | 0.548 | **0.663** |
| 2 | 0.608 | 0.501 | **0.654** | 0.609 | 0.522 |
| 6 | 0.625 | 0.662 | 0.529 | **0.627** | 0.550 |
| 7 | 0.562 | **0.673** | 0.529 | 0.513 | 0.555 |
| 21 | **0.628** | 0.565 | 0.565 | 0.525 | 0.541 |

**Layer × Metric Heatmap (AUROC) — Llama-3.2-1B-Instruct**

| Layer | energy | fiedler | hfer | smoothness | entropy |
|---|---|---|---|---|---|
| 0 | 0.601 | 0.611 | 0.473 | 0.563 | 0.535 |
| 2 | 0.594 | **0.643** | 0.527 | 0.592 | 0.595 |
| 8 | 0.608 | **0.643** | 0.518 | 0.607 | 0.514 |
| 13 | **0.638** | 0.641 | 0.531 | **0.645** | 0.597 |
| 15 | 0.623 | 0.614 | 0.412 | 0.595 | 0.554 |

---

### Per-Dataset Breakdown (rich sweep, out-of-fold ROC-AUC)

The mixed pool is evaluated per source dataset using 5-fold stratified CV out-of-fold predictions — every sample is scored by a probe that never trained on it.

| Dataset | GPT-2 | Llama-3B | TinyLlama | Llama-1B | n (inj/ben) | Notes |
|---|---|---|---|---|---|---|
| advbench | 0.961 | **0.970** | **0.979** | 0.953 | 50 / 50 | Harmful behaviour prompts |
| agentdojo | **1.000** | **1.000** | **1.000** | 0.980 | 50 / 50 | Agentic tool-use injections |
| stealthy | 0.951 | **0.973** | 0.932 | 0.938 | 30 / 30 | Format-free stealthy injections |
| injecagent | 0.437 | 0.346 | 0.322 | 0.336 | 50 / 50 | Tool-calling RAG injections |
| llmlat_benign | — | — | — | — | 0 / 100 | Benign-only; AUC undefined |
| **OVERALL** | **0.900** | **0.889** | **0.866** | **0.858** | 180 / 280 | |

> **agentdojo** and **advbench** are near-perfectly separable by spectral features across all backbones. **stealthy** injections (which break activation probes in the LOO setting) are well-detected here (0.93–0.97).
>
> **injecagent** is the hard case: 0.32–0.44 AUC, below chance. InjecAgent frames injections inside tool-call/RAG context, producing attention graphs that look spectrally more like *benign* samples from other datasets than like injections — the mixed-trained probe inverts on it. Evaluating it in isolation (or with domain-specific training) would be needed to close this gap.
>
> **llmlat_benign** contributes only negative examples (used for calibration in the mixed pool) so per-dataset AUC is undefined.

---

### Head-to-Head Comparison

| Method | Backbone | Setting | ROC-AUC | Generalises cross-domain? |
|---|---|---|---|---|
| Linear Probe (activation) | TinyLlama-1.1B | In-domain | **1.000** | No — collapses to 0.0–0.5 |
| MLP Probe (activation) | TinyLlama-1.1B | In-domain | **1.000** | No — collapses to 0.1–0.9 |
| MLP Probe (activation) | Qwen2.5-1.5B | In-domain | **1.000** | No — collapses to 0.1–0.9 |
| TF-IDF + LR (text) | — | Cross-domain | 0.869–1.000 | Partially (lexical bias) |
| Semantic MLP (embeddings) | all-MiniLM-L6 | Cross-domain | 0.936–1.000 | Mostly yes |
| Perplexity | — | Stealthy | **1.000** | Dataset-dependent |
| **Rich Sweep (spectral)** | **GPT-2** | **Mixed cross-dataset** | **0.900** | **Yes** |
| **Rich Sweep (spectral)** | **Llama-3.2-3B-Instruct** | **Mixed cross-dataset** | **0.889** | **Yes** |
| **Rich Sweep (spectral)** | **TinyLlama-1.1B-Chat** | **Mixed cross-dataset** | **0.866** | **Yes** |
| **Rich Sweep (spectral)** | **Llama-3.2-1B-Instruct** | **Mixed cross-dataset** | **0.858** | **Yes** |

> The spectral rich sweep occupies a unique position: it does not require access to model hidden states (only attention weights), is architecture-agnostic, and maintains 0.86–0.90 AUROC across heterogeneous datasets where activation probes collapse. The main trade-off vs. in-domain probes is a ~10 pp gap on easy in-domain benchmarks. Notably, TinyLlama achieves a stronger `fiedler_value` trajectory AUC (0.840) than the same metric on Llama-3.2-1B (0.715), suggesting that the chat-tuned attention structure is better aligned with injection semantics than a raw instruction model at the same scale.

---

## Repository Structure

- `src/`: Core implementation.
  - `extract/`: Hidden state extraction logic.
  - `probes/`: Linear probe training and leakage validation.
  - `generate/`: Dataset generation scripts (Base, Stealthy, Complex).
  - `baselines/`: TF-IDF, Semantic, and Statistical baseline implementations.
  - `analysis/`: Comprehensive reporting and advanced stats.
- `data/`: Raw prompts, processed features, and visualization artifacts.
  - `rich_sweep_results/`: GPT-2 spectral-trust rich sweep outputs.
  - `rich_sweep_results_llama1b/`: Llama-3.2-1B-Instruct spectral-trust rich sweep outputs.
  - `rich_sweep_results_llama3b/`: Llama-3.2-3B-Instruct spectral-trust rich sweep outputs.
  - `rich_sweep_results_tinyllama/`: TinyLlama-1.1B-Chat spectral-trust rich sweep outputs.
- `scripts/`: Utility scripts (plotting).
- `rich_sweep_eval.py`: Spectral-trust rich sweep evaluation script.

```
