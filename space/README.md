---
title: Peptide Detectability Predictor
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

<!-- # Peptide Detectability Predictor
Predicts whether a peptide is likely to be detected ("flyer") in a bottom-up
mass-spectrometry experiment. Inspired by Saeed Lab (FIU) research on recovering
overlooked peptides in MS-based omics. -->

This folder is a **self-contained Gradio Space**: `app.py` is the entry point and
`src/` is a small inference-only package. Trained models live in `models/`
(`registry.json`, `LATEST.json`, and one directory per version). See
`prepare_models.py` to copy them in from the training repo before pushing.
