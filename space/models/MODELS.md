# space/models/

The Gradio app loads models from **this** folder — paths resolve relative to
`space/` as the repo root (`space/models/registry.json`, `space/models/LATEST.json`,
and one directory per version).

It is empty in the repo because models are produced by **training on your machine**.
After training locally, populate it — either with the helper:

```bash
cd space
python prepare_models.py                 # registry + LATEST + all versions
# or: python prepare_models.py --champion-only
# or: python prepare_models.py --versions v001_baseline_... v003_esm2_...
```

…or copy manually from the repo root:

```bash
cp ../models/registry.json ../models/LATEST.json ./
cp -r ../models/v001_baseline_* ./       # and any other versions you want selectable
```

Then push the Space (build guide, Phase 8). At minimum, include the **champion**.
`*.pt` and `*.pkl` are tracked via Git LFS (see `../.gitattributes`).
