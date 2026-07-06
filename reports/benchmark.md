# Benchmark

Evaluated on the same held-out test split (`data/processed/test.csv`, n=3692, positive rate=50.5%). Ranked by PR-AUC (descending). Primary metric: **pr_auc**.

| version | model_type | param_count | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v001_baseline_20260704_013915 ★ | baseline | 2,480 | 0.6888 | 0.6604 | 0.7897 | 0.7193 | 0.7408 | 0.7026 |

★ = current champion (highest PR-AUC, per `models/LATEST.json`).

## Figures

![ROC curves](figures/roc.png)

![Precision–Recall curves](figures/pr.png)

![Model comparison](figures/model_comparison.png)

> **Note.** `param_count` is the number of trainable parameters for the neural models (cnn, esm2) and the total number of leaves across all trees for the LightGBM baseline.

> **Footnote — context, not a head-to-head comparison.** The DeepMSPeptide paper (Serrano et al., *Bioinformatics* 2020) reported roughly **93% sensitivity / 99% specificity** on **their own** test set. This benchmark uses a *different* train/val/test split of the GPMDB-derived data, so those figures are **context only** and are **not** directly comparable to the numbers above.
