---
title: Peptide Detectability Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.25.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predict peptide detectability in bottom-up MS proteomics
tags:
  - proteomics
  - mass-spectrometry
  - bioinformatics
  - peptides
---

# 🧬 Peptide Detectability Predictor

Predicts whether a tryptic peptide is likely to be **detected** (a proteotypic
"flyer") in a bottom-up mass-spectrometry experiment, or likely to be
**overlooked**. Useful for choosing proteotypic peptides for targeted assays
or for spotting peptides that a discovery run is unlikely to see.

**Research use only.** Predictions are a screening aid, not a guarantee of
detection on any particular instrument or workflow.

## Using the app

| Tab | Input | Output |
|---|---|---|
| Single peptide | one sequence | P(detectable) and a verdict |
| Batch (paste) | one sequence per line, up to 5,000 | sortable table + CSV download |
| CSV upload | CSV with a `sequence` column (else the first column), up to 5,000 rows / 5 MB | sortable table + CSV download |

**Accepted input:** unmodified peptides of **7–50 residues** over the 20
standard amino acids (`ACDEFGHIKLMNPQRSTVWY`), matching the training data.
Whitespace and case are ignored. Sequences with other characters (`X`, `B`,
`U`, modification notation such as `M(ox)`) or outside the length range are
listed as *not scored* rather than given a probability.

**Verdict:** P(detectable) ≥ 0.5 → *likely detectable*; below → *likely
OVERLOOKED*.

## Model

LightGBM gradient-boosted trees on 27 hand-crafted features: length, the 20
amino-acid fractions, and six Biopython ProtParam descriptors (molecular
weight, GRAVY, aromaticity, instability index, isoelectric point, net charge
at pH 7).

Held-out test set (n = 3,692, 50.5% positive):

| PR-AUC | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.703 | 0.741 | 0.689 | 0.660 | 0.790 | 0.719 |

Full report with ROC / PR curves: [`reports/benchmark.md`](reports/benchmark.md).

### Limitations

- **Training data.** 24,612 unique peptides (17,228 for training) from the
  DeepMSPeptide GPMDB table `GPMDB_test_peptides.txt`. The larger
  `GPMDB_training_peptides.rar` set was not used for this version.
- **Labels** come from GPMDB observation frequency (MObs → detectable,
  LObs → not detectable). They reflect what was historically observed across
  many public experiments, not detectability on a specific instrument,
  digestion protocol or sample type.
- **Not comparable to the DeepMSPeptide paper.** That paper reports ~93%
  sensitivity / 99% specificity on its own split; this project uses a
  different, leakage-free 70/15/15 split, so the numbers cannot be compared
  directly.
- Modified residues, non-tryptic peptides and lengths outside 7–50 are out of
  scope.

## Running locally

The project has two front ends over the same model and input rules
(`src/validation.py`):

| Front end | Entry point | Hosted on | Run locally |
|---|---|---|---|
| Gradio app | `app.py` | Hugging Face Spaces | `python app.py` → <http://127.0.0.1:7860> |
| Web page + JSON API | `index.py` + `web/` | Vercel | `uvicorn index:app` → <http://127.0.0.1:8000> |

```bash
pip install -r requirements.txt gradio==6.25.0 uvicorn
```

### JSON API

```bash
curl -X POST https://<your-deployment>/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["TITLEVEPSDTIENVK", "HQGVMVGMGQK"]}'
```

Returns one result per sequence (`probability` is `null` when the sequence is
outside the model's domain). `GET /api/model` describes the model and input
limits; interactive docs are at `/api/docs`.

## Deploying

**Hugging Face Spaces** — push this repository to the Space; it runs `app.py`
and installs Gradio from `sdk_version` above.

**Vercel** (Hobby plan is free for non-commercial use):

1. Import the GitHub repository at <https://vercel.com/new>. Vercel detects
   FastAPI from `requirements.txt`; no build settings are needed.
2. In **Settings → Git**, turn on **Git LFS** (the model `.pkl` files are stored
   with LFS) and set the production branch to `main`.
3. Deploy. `.vercelignore` limits the upload to `index.py`, `src/`, `models/`,
   `web/` and the config files, so the Gradio `app.py` is not picked up.

Limits on Vercel: 5,000 sequences per request and 4 MB per CSV upload (request
bodies are capped at 4.5 MB). The first request after an idle period takes a
few seconds while the function starts.

## Reproducing / training

```bash
pip install -r requirements-train.txt
python -m src.data_prep               # download + clean + split (see data/README.md)
python -m src.train --model baseline  # also: cnn, esm2
python -m src.benchmark               # writes reports/benchmark.md + figures
pytest
```

Every training run is saved as a new, never-overwritten version under
`models/` and recorded in `models/registry.json`. The best by PR-AUC becomes the
champion in `models/LATEST.json`, which the app selects by default. Data
sourcing, cleaning and split details are in [`data/README.md`](data/README.md).
ESM-2 embeddings can be precomputed on a free Colab GPU with
[`notebooks/colab_esm2_embeddings.ipynb`](notebooks/colab_esm2_embeddings.ipynb).

```
app.py                Gradio app (Hugging Face Space entrypoint)
index.py              FastAPI JSON API (Vercel entrypoint)
web/                  static page served by index.py
src/                  data prep, features, models, training, evaluation, inference,
                      shared input validation (validation.py)
models/               versioned model artifacts (Git LFS) + registry
reports/              benchmark report and figures
data/README.md        dataset documentation
tests/                pytest suite
```

## Citation

If you use this tool, please cite the data sources:

> Serrano G, Guruceaga E, Segura V. **DeepMSPeptide: peptide detectability
> prediction using deep learning.** *Bioinformatics.* 2020;36(4):1279–1280.
> doi:[10.1093/bioinformatics/btz708](https://doi.org/10.1093/bioinformatics/btz708)

> Craig R, Cortens JP, Beavis RC. **Open source system for analyzing,
> validating, and storing protein identification data.** *J Proteome Res.*
> 2004;3(6):1234–1242. doi:[10.1021/pr049882h](https://doi.org/10.1021/pr049882h)

## License

Code and models: [MIT](LICENSE). The DeepMSPeptide dataset is MIT-licensed by
its authors.
