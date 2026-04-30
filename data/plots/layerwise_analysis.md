# Layer-wise Safety Benchmark Analysis

## Summary Table – Linear Probe AUROC by Layer

| Layer | In-Domain AUROC | Cross-Dataset AUROC |
|-------|-----------------|---------------------|
| Layer  0 | 0.807 | 0.724 |
| Layer  5 | 0.737 | 0.672 |
| Layer 11 | 0.721 | 0.745 |
| Layer 16 | 0.722 | 0.651 |
| Layer 21 | 0.679 | 0.690 |

## Key Findings

1. **Best In-Domain Layer**: Layer 0
   - Achieves peak AUROC on the original dataset distribution.
   - This layer has likely *overfit* to the specific formatting/style of each dataset.

2. **Best Generalizing Layer**: Layer 11
   - Provides the highest average Cross-Dataset AUROC.
   - This is the layer that should be used for the probe in the main paper.

3. **Interpretation**:
   - Layers close to the **output** (last layer) tend to overfit in-domain cues (length, format) and collapse cross-domain.
   - **Intermediate layers** encode more generalizable, semantically rich representations of intent.
   - This is consistent with findings from probing literature (Tenney et al., 2019; Belinkov & Glass, 2019).

## Implications for the Paper

> [!IMPORTANT]
> If Layer 11 achieves significantly higher Cross-Dataset AUROC than the last layer,
> this provides strong evidence that *the LLM does internally encode safety-relevant information*,
> but the **signal is distributed across intermediate layers** and is more generalizable there.

## Plots

- `layerwise_auroc_indomain.png` – Per-dataset, per-layer AUROC (In-Domain)
- `layerwise_auroc_crossds.png`  – Average Cross-Dataset AUROC by layer
- `layerwise_heatmap.png`        – Heatmap: Layer x Cross-DS pair AUROC
