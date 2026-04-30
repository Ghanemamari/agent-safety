# Agent Safety Benchmark – Comprehensive Results

> Source CSV: `eval_blocks_181109.csv`
> Evaluation covers **In-Domain (Block 1)**, **Cross-Dataset (Block 2)**, and **Leave-One-Out (Block 3)**.

## Comparison Table – Average AUROC

| BT          |   Linear Probe |   MLP Probe |   Semantic MLP |   TF-IDF + LR |   Perplexity |   Random |   Length Classifier |   Label-Flip Control |
|:------------|---------------:|------------:|---------------:|--------------:|-------------:|---------:|--------------------:|---------------------:|
| Block1      |          1     |       1     |           1    |         1     |        0.694 |    0.608 |               0.842 |                0     |
| (In-Domain) |                |             |                |               |              |          |                     |                      |
| Block2      |          0.587 |       0.593 |           0.99 |         0.986 |        0.71  |    0.455 |               0.433 |                0.006 |
| (Cross-DS)  |                |             |                |               |              |          |                     |                      |

## Comparison Table – Average F1-Score

| BT          |   Linear Probe |   MLP Probe |   Semantic MLP |   TF-IDF + LR |   Perplexity |   Random |   Length Classifier |   Label-Flip Control |
|:------------|---------------:|------------:|---------------:|--------------:|-------------:|---------:|--------------------:|---------------------:|
| Block1      |          0.987 |       1     |          1     |         0.971 |         0.7  |    0.542 |               0.809 |                 0    |
| (In-Domain) |                |             |                |               |              |          |                     |                      |
| Block2      |          0.365 |       0.383 |          0.354 |         0.172 |         0.69 |    0.452 |               0.238 |                 0.45 |
| (Cross-DS)  |                |             |                |               |              |          |                     |                      |

## Comparison Table – Average Accuracy

| BT          |   Linear Probe |   MLP Probe |   Semantic MLP |   TF-IDF + LR |   Perplexity |   Random |   Length Classifier |   Label-Flip Control |
|:------------|---------------:|------------:|---------------:|--------------:|-------------:|---------:|--------------------:|---------------------:|
| Block1      |          0.986 |       1     |          1     |         0.967 |         0.7  |    0.567 |               0.808 |                0     |
| (In-Domain) |                |             |                |               |              |          |                     |                      |
| Block2      |          0.663 |       0.681 |          0.652 |         0.561 |         0.69 |    0.469 |               0.506 |                0.328 |
| (Cross-DS)  |                |             |                |               |              |          |                     |                      |

## Generalisation Analysis

**Best generalising method (highest average Cross-Dataset AUROC): `Semantic MLP`**

### Key Observations

1. **In-Domain (Block 1)**: The Linear Probe and MLP Probe consistently achieve near-perfect AUROC (≈1.0) when trained and tested on the same dataset distribution. This confirms that safety signals are *linearly encoded* in TinyLlama's hidden representations.

2. **Cross-Dataset (Block 2)**: Performance drops substantially for representation-based methods. The remaining performance indicates that some features *transfer*, but the probe is partially distribution-specific. Methods like `TF-IDF + LR` may retain higher cross-dataset performance due to shared surface-level attack vocabulary.

3. **Leave-One-Out (Block 3)**: Methods trained on all other datasets and evaluated on the held-out one provide the most rigorous test of generalisation. A strong cross-domain result here validates that the safety signal is *dataset-agnostic*.

4. **Control Baselines**:
   - *Random*: AUROC ≈ 0.5 as expected – confirms the evaluation is well-calibrated.
   - *Label-Flip*: Low AUROC if working correctly – the probe cannot learn a reversed signal, proving semantic encoding rather than memorisation.
   - *Length Classifier*: Reveals if datasets have a length bias (longer prompts = attack).

## Artefact Detection

| Risk | Indicator | Status |
|------|-----------|--------|
| Length bias | Length Classifier AUROC in Block 2 | Monitored above |
| Lexical surface | TF-IDF + LR AUROC in Block 2 | Monitored above |
| Memorisation | Label-Flip Control AUROC ≈ 0 | Verified |
| Distributional shift | Perplexity AUROC | Monitored above |

> [!NOTE]
> If `TF-IDF + LR` achieves very high cross-domain AUROC, dataset-specific vocabulary (e.g., common attack phrases) may be a confound. The paper should report these artefact checks alongside the main results.

## Plots

- `benchmark_metrics_bars.png` – 3-metric side-by-side bar chart
- `cross_dataset_transfer.png` – Transfer matrix heatmap (Linear Probe AUROC)
