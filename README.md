---
title: Peptide Detectability
emoji: 🦀
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.25.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Data

## Source

Peptides come from the public **DeepMSPeptide** dataset, which in turn derives
its peptides from the **GPMDB** (Global Proteome Machine Database) — a record of
peptides observed across hundreds of MS/MS experiments together with how often
each was detected.

- Repository: <https://github.com/vsegurar/DeepMSPeptide>
- Upstream license: **MIT** (Copyright © 2019 Guillermo Serrano Sanz) — see the
  repository's `LICENSE`. Peptides are re-derived here only for research /
  benchmarking; cite the paper and GPMDB below, and check the upstream license
  before redistributing any data.

### Files used (downloaded into `data/raw/`, which is git-ignored)

| File | Format | Contents |
|------|--------|----------|
| `GPMDB_test_peptides.txt` | TSV, ~25k rows | `Peptide_seq` + ~57 physicochemical columns + `Class` |
| `GPMDB_training_peptides.rar` | RAR → TSV | same schema; **optional**, needs a RAR extractor |

We use only the peptide **sequence** (`Peptide_seq`) and the **`Class`** column.
The pre-computed physicochemical columns in these files are ignored — we
recompute our own features from scratch in `src/features.py` (Phase 3).

### Label convention

The `Class` column encodes GPMDB observation frequency, mapped to a binary
detectability label:

| `Class` | Meaning | Label |
|---------|---------|:-----:|
| `MObs`  | multiply observed / proteotypic | **1 — detectable** |
| `LObs`  | lowly observed | **0 — not detectable** |

## Processing (`python -m src.data_prep`)

1. **Download** the GPMDB tables into `data/raw/` (via `urllib`, no extra deps).
2. **Clean** — uppercase; keep only sequences written over the 20 standard amino
   acids (`ACDEFGHIKLMNPQRSTVWY`); keep length in `[MIN_LEN, MAX_LEN] = [7, 50]`.
3. **Deduplicate *before* splitting** — collapse exact-duplicate sequences and
   **drop any sequence that appears with conflicting labels**, so no peptide can
   land in more than one split (this prevents train/test leakage).
4. **Stratified 70/15/15 split** (`RANDOM_SEED = 42`) into
   `data/processed/{train,val,test}.csv`, each with columns `[sequence, label]`.
5. Write `data/processed/split_stats.json` — per-split row counts and class
   balance, plus a 12-character SHA-256 hash of the three CSVs (`data_hash`).

Both `data/raw/` and `data/processed/` are **git-ignored** and are regenerated
in full by the command above.

> **Note on the split.** We create our own 70/15/15 split from the pooled,
> deduplicated peptides rather than reusing the repository's train/test files.
> Results here are therefore **not** directly comparable to the DeepMSPeptide
> paper's reported numbers (which used a different split).

### The training set is a `.rar`

`GPMDB_test_peptides.txt` is plain text and already contains both classes in
near-equal proportion, so the pipeline runs on it directly. To *also* include
`GPMDB_training_peptides.rar`, install a RAR extractor — one of `unrar`, `unar`,
`libarchive-tools` (`bsdtar`), or `p7zip` (optionally `pip install rarfile`) —
and rerun. `src/data_prep.py` picks up the extracted table automatically.

### Offline / blocked network

If the download is blocked, `src/data_prep.py` prints a warning and generates a
small, clearly-labelled **synthetic placeholder** dataset (marked
`"synthetic": true` in `split_stats.json`) so the rest of the code can run
end-to-end. Force it with `python -m src.data_prep --synthetic`. Replace it with
real data before reporting any results.

## Citations

If you use this data, please cite the DeepMSPeptide paper and GPMDB:

> Serrano G, Guruceaga E, Segura V. **DeepMSPeptide: peptide detectability
> prediction using deep learning.** *Bioinformatics.* 2020;36(4):1279–1280.
> doi:[10.1093/bioinformatics/btz708](https://doi.org/10.1093/bioinformatics/btz708)

> Craig R, Cortens JP, Beavis RC. **Open source system for analyzing, validating,
> and storing protein identification data.** *J Proteome Res.* 2004;3(6):1234–1242.
> doi:[10.1021/pr049882h](https://doi.org/10.1021/pr049882h)
