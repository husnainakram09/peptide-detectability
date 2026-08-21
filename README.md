---
title: Peptide Detectability Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Peptide Detectability Predictor

A small, honest, fully reproducible machine-learning project that predicts **peptide
detectability** — whether a given amino-acid sequence is likely to be *observed*
(a proteotypic "flyer" peptide) in a bottom-up mass-spectrometry experiment, or
likely to be **overlooked**. It reproduces the DeepMSPeptide baseline idea and
benchmarks it head-to-head against a 1D-CNN and a small protein-language-model
(ESM-2) head on a single shared test split, with a never-overwrite model registry
so no experiment clobbers another.

- **Live demo (Hugging Face Space):** `https://huggingface.co/spaces/Husnainakram/peptide-detectability`
  &nbsp;— *replace with your Space URL after deploying (see “Deploy”).*
- **What this is:** a faithful reproduction + a fair three-way benchmark + an ESM-2
  extension. **What it is not:** novel research, or a claim to “beat” any published model.

---

## Scientific motivation

In bottom-up proteomics, a protein is digested into peptides that are then measured
by the mass spectrometer. Only a **subset** of a protein’s peptides are ever
detected — the *proteotypic* or "flyer" peptides — because peptide selection in the
instrument is effectively stochastic and depends heavily on each peptide’s
amino-acid sequence and physicochemical properties. The peptides that are **not**
detected are the *overlooked* peptides.

A model that predicts detectability is, by construction, a model of **which peptides
get missed**. Framing detectability as the flip side of the overlooked-peptide
problem is what connects this small project to the broader research theme of
recovering overlooked peptides in MS-based omics (see “Inspired by”).

---

## Architecture — three models, one test split

All three models train on the same data and are evaluated on the same held-out test
split. Reporting all three (and being honest about which wins) is the point.

| Model | Learns from | Why it’s here | CPU cost |
|---|---|---|---|
| **Baseline** — LightGBM | engineered features: length, 20-AA composition, and physicochemical descriptors (molecular weight, GRAVY, aromaticity, instability index, isoelectric point, net charge at pH 7) | a strong, fast, interpretable reference that mirrors classic detectability approaches | seconds |
| **1D-CNN** (DeepMSPeptide-style) | the raw sequence, integer-encoded | learns local sequence motifs directly, no hand-engineered features | a few minutes |
| **ESM-2 + MLP head** | frozen embeddings from a small pretrained protein language model (`facebook/esm2_t6_8M_UR50D`) | tests whether large-scale pretraining helps on short peptides; embeddings are cached once, then a tiny head trains in seconds | minutes (embeddings cached once; extract on Colab or CPU) |

A benchmark ties them together: one comparison table, overlaid ROC and
precision–recall curves, and a PR-AUC/F1 bar chart.

Detectability datasets are typically **class-imbalanced**, so the **primary metric is
PR-AUC** (not accuracy), and every model uses class weighting.

---

## Quickstart / reproduce it

```bash
# 1) environment (Python 3.11)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if torch didn't install the CPU build

# 2) data: download + clean + dedup + stratified 70/15/15 split
python -m src.data_prep
#   -> data/processed/{train,val,test}.csv + split_stats.json
#   (blocked network? it falls back to a clearly-labelled synthetic placeholder;
#    the training set ships as a .rar — install unrar/unar/libarchive-tools/p7zip
#    to include it, otherwise the plain-text test table is used on its own.)

# 3) train all three models, then benchmark
bash scripts/train_all.sh
#   == python -m src.data_prep
#      python -m src.train --model baseline     # -> models/v001_baseline_*
#      python -m src.train --model cnn          # -> models/v002_cnn_*
#      python -m src.train --model esm2         # -> models/v003_esm2_*
#      python -m src.benchmark                  # -> reports/benchmark.md + figures

# 4) try the demo locally
python app/app.py
```

**Individual commands**

```bash
python -m src.train --model baseline           # or: cnn, esm2 (esm2: --esm-model ...)
python -m src.evaluate --version v002_cnn_...   # metrics for one saved version
python -m src.benchmark                         # compare all versions on the same test split
```

**ESM-2 embeddings (optional speed-up).** Extracting ESM-2 embeddings is the only
step that benefits from a GPU. Run `notebooks/colab_esm2_embeddings.ipynb` on a free
Colab GPU, download the resulting `.npy` files into `embeddings_cache/`, and
`python -m src.train --model esm2` will load them instead of recomputing. (It also
works on CPU, just slower.)

---

## Model versioning — nothing is ever overwritten

Every training run creates a **brand-new, timestamped, index-incremented** directory
under `models/` (e.g. `v001_baseline_20250101_120000`). The registry is
**append-only** and the champion pointer can change, but **no model is ever deleted
or overwritten**. This is enforced in `src/versioning.py`:

- `new_version_dir(model_type)` always makes a *new* folder (and fails loudly rather
  than reuse one), so re-running training as many times as you like keeps every prior
  model intact.
- `register(...)` appends an entry (version, model type, git commit, data hash,
  metrics, config) to `models/registry.json` and writes `metrics.json` / `config.json`
  inside the version directory.
- `update_champion(metric="pr_auc")` recomputes `models/LATEST.json` to point at the
  best model by PR-AUC — **deleting nothing**.

The unit tests in `tests/test_versioning.py` assert this never-overwrite,
append-only contract.

---

## Results

Run `python -m src.benchmark` to (re)generate `reports/benchmark.md` and the figures
below from the **committed models** — everything is reproducible from the saved
versions plus the regenerated test split. Then paste the generated table here.

<!-- Replace the row cells below with the output of reports/benchmark.md. -->

| version | model_type | param_count | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| … | baseline | … | … | … | … | … | … | … |
| … | cnn | … | … | … | … | … | … | … |
| … | esm2 | … | … | … | … | … | … | … |

**Champion:** _the highest-PR-AUC model, recorded in `models/LATEST.json` (marked
with ★ in `reports/benchmark.md`)._ Update this line with your actual winner.

![ROC curves](reports/figures/roc.png)
![Precision–Recall curves](reports/figures/pr.png)
![Model comparison](reports/figures/model_comparison.png)

> **Context, not a head-to-head comparison.** The DeepMSPeptide paper reported
> roughly **93% sensitivity / 99% specificity** on **its own** test set. This project
> uses a *different* train/val/test split of the GPMDB-derived data, so those figures
> are **context only** and are **not** directly comparable to the numbers above.

---

## Repository layout

```
peptide-detectability/
├── src/
│   ├── config.py          # paths, amino-acid vocab, constants, primary metric
│   ├── data_prep.py       # download → clean → dedup → stratified split → hash
│   ├── features.py        # composition + physicochemical features (baseline)
│   ├── encoders.py        # integer encoding (CNN) + ESM-2 embeddings + cache
│   ├── versioning.py      # append-only, never-overwrite model registry
│   ├── train.py           # entrypoint: --model {baseline,cnn,esm2}
│   ├── evaluate.py        # metrics for any saved version
│   ├── benchmark.py       # compare all versions → table + figures
│   ├── predict.py         # load a version/champion → predict on new peptides
│   └── models/            # baseline.py, cnn.py, esm_head.py
├── app/                   # Gradio demo (app.py) + HF Space front-matter
├── space/                 # self-contained folder ready to push to a HF Space
├── notebooks/             # colab_esm2_embeddings.ipynb (optional GPU extraction)
├── data/                  # provenance + citations; raw/ & processed/ (git-ignored)
├── models/                # versioned models + registry.json + LATEST.json
├── reports/               # benchmark.md + figures/
└── tests/                 # test_versioning.py, test_data_prep.py
```

---

## Deploy (free, Hugging Face Spaces)

`space/` is a self-contained Gradio Space (its own `app.py`, `src/` package,
`requirements.txt`, and `README.md`). After training, copy your models in and push:

```bash
cd space
python prepare_models.py --champion-only     # or: python prepare_models.py
# create a Gradio Space (CPU basic, free) on huggingface.co, then:
git lfs install && git add . && git commit -m "Deploy peptide detectability demo" && git push
```

`*.pt`/`*.pkl` are tracked via Git LFS (`space/.gitattributes`). Put the resulting
public URL at the top of this README.

---

## Data provenance & citations

Peptides come from the public **DeepMSPeptide** dataset, derived from **GPMDB**
(the Global Proteome Machine Database). In the source tables the `Class` column
(GPMDB observation frequency) is mapped to a binary label: `MObs` (multiply observed
/ proteotypic) → **1 = detectable**, `LObs` (lowly observed) → **0 = not detectable**.
Full details, the label convention, and licensing are in
[`data/README.md`](data/README.md). Upstream license: MIT (© 2019 Guillermo Serrano
Sanz) — check it before redistributing any data.

> Serrano G, Guruceaga E, Segura V. **DeepMSPeptide: peptide detectability prediction
> using deep learning.** *Bioinformatics.* 2020;36(4):1279–1280.
> doi:[10.1093/bioinformatics/btz708](https://doi.org/10.1093/bioinformatics/btz708)

> Craig R, Cortens JP, Beavis RC. **Open source system for analyzing, validating, and
> storing protein identification data (GPMDB).** *J Proteome Res.* 2004;3(6):1234–1242.
> doi:[10.1021/pr049882h](https://doi.org/10.1021/pr049882h)

---

## License

MIT — see [`LICENSE`](LICENSE).
