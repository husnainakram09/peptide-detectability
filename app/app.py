"""Gradio demo for the peptide detectability predictor (inference only).

Tabs:
  1. Single peptide  -> P(detectable) + verdict (likely detectable / OVERLOOKED)
  2. Batch (paste)   -> sortable results table + Download CSV
  3. CSV upload      -> same table + Download CSV

A dropdown selects the model version (populated from models/registry.json),
defaulting to the champion. The app never trains and is CPU-only.

Run locally:  python app/app.py
"""
from __future__ import annotations

import os
import sys
import tempfile

# Make ``from src import ...`` importable when running ``python app/app.py`` from
# the repo root (repo root = parent of app/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import gradio as gr  # noqa: E402

from src import predict as predictor  # noqa: E402

THRESHOLD = 0.5
_COLUMNS = ["sequence", "probability", "verdict"]
_CACHE: dict = {}  # version name -> predict callable (loaded once)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _versions() -> list:
    try:
        return [e["version"] for e in predictor.list_versions()]
    except Exception:
        return []


def _default_version():
    champ = predictor.champion_version()
    vs = _versions()
    if champ and champ in vs:
        return champ
    return vs[-1] if vs else None


def _entries() -> dict:
    try:
        return {e["version"]: e for e in predictor.list_versions()}
    except Exception:
        return {}


def _model_info_md(version) -> str:
    if not version:
        return ("**No model available.** Train one first, e.g. "
                "`python -m src.train --model baseline`.")
    entry = _entries().get(version, {})
    champ = predictor.champion_version()
    star = " — ★ champion" if version == champ else ""
    metrics = entry.get("metrics", {})
    bits = []
    for key in ("pr_auc", "roc_auc", "f1"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            bits.append(f"{key}={val:.3f}")
    metric_str = ("  |  " + ", ".join(bits)) if bits else ""
    return f"**Active model:** `{version}`  (type: {entry.get('model_type', '?')}){star}{metric_str}"


def _get_predictor(version):
    if version not in _CACHE:
        _CACHE[version] = predictor.load_model(version)
    return _CACHE[version]


def _clean_seq(s) -> str:
    return "".join(ch for ch in str(s).strip().upper() if ch.isalpha())


def _verdict(p: float) -> str:
    return "likely detectable" if p >= THRESHOLD else "likely OVERLOOKED"


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def _predict_frame(sequences, version):
    """Return (results_dataframe, csv_filepath_or_None)."""
    if not version:
        return _empty_df(), None
    seqs = [_clean_seq(s) for s in sequences]
    seqs = [s for s in seqs if s]
    if not seqs:
        return _empty_df(), None
    probs = _get_predictor(version)(seqs)
    df = pd.DataFrame({
        "sequence": seqs,
        "probability": [round(float(p), 4) for p in probs],
        "verdict": [_verdict(float(p)) for p in probs],
    })
    tmp = tempfile.NamedTemporaryFile(prefix="peptide_predictions_", suffix=".csv", delete=False)
    tmp.close()
    df.to_csv(tmp.name, index=False)
    return df, tmp.name


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def predict_single(sequence, version):
    if not version:
        return "No model available — train one first."
    seq = _clean_seq(sequence)
    if not seq:
        return "Enter a peptide sequence (letters only)."
    prob = float(_get_predictor(version)([seq])[0])
    mark = "✅" if prob >= THRESHOLD else "🚫"
    return (f"### {mark} {_verdict(prob)}\n\n"
            f"**P(detectable) = {prob:.3f}**  for `{seq}`  (length {len(seq)})")


def predict_batch(text, version):
    lines = (text or "").splitlines()
    return _predict_frame(lines, version)


def predict_csv(file, version):
    if file is None:
        return _empty_df(), None
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path:
        return _empty_df(), None
    df_in = pd.read_csv(path)
    col = next((c for c in df_in.columns if str(c).strip().lower() == "sequence"), None)
    if col is None:
        col = df_in.columns[0]  # fall back to the first column
    return _predict_frame(df_in[col].tolist(), version)


def refresh_models():
    vs = _versions()
    default = _default_version()
    return gr.update(choices=vs, value=default), _model_info_md(default)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_app() -> gr.Blocks:
    versions = _versions()
    default = _default_version()

    with gr.Blocks(title="Peptide Detectability Predictor") as demo:
        gr.Markdown(
            "# 🧬 Peptide Detectability Predictor\n"
            "Predicts whether a peptide is likely to be **detected** (a proteotypic "
            "\u201cflyer\u201d) in a bottom-up mass-spectrometry experiment, or likely "
            "**overlooked**. Inference only — CPU-friendly."
        )

        with gr.Row():
            version_dd = gr.Dropdown(choices=versions, value=default,
                                     label="Model version", scale=4)
            refresh_btn = gr.Button("🔄 Refresh models", scale=1)
        active_md = gr.Markdown(_model_info_md(default))
        version_dd.change(_model_info_md, inputs=version_dd, outputs=active_md)
        refresh_btn.click(refresh_models, outputs=[version_dd, active_md])

        with gr.Tab("Single peptide"):
            seq_in = gr.Textbox(label="Peptide sequence",
                                placeholder="e.g. TITLEVEPSDTIENVK")
            single_btn = gr.Button("Predict", variant="primary")
            single_out = gr.Markdown()
            single_btn.click(predict_single, inputs=[seq_in, version_dd], outputs=single_out)
            seq_in.submit(predict_single, inputs=[seq_in, version_dd], outputs=single_out)

        with gr.Tab("Batch (paste)"):
            multi_in = gr.Textbox(label="One peptide per line", lines=8,
                                  placeholder="TITLEVEPSDTIENVK\nHQGVMVGMGQK\nESTLHLVLR")
            batch_btn = gr.Button("Predict batch", variant="primary")
            batch_tbl = gr.Dataframe(headers=_COLUMNS, interactive=False, wrap=True,
                                     label="Results (click a column header to sort)")
            batch_file = gr.File(label="Download CSV")
            batch_btn.click(predict_batch, inputs=[multi_in, version_dd],
                            outputs=[batch_tbl, batch_file])

        with gr.Tab("CSV upload"):
            csv_in = gr.File(label="CSV with a 'sequence' column",
                             file_types=[".csv"], type="filepath")
            csv_btn = gr.Button("Predict from CSV", variant="primary")
            csv_tbl = gr.Dataframe(headers=_COLUMNS, interactive=False, wrap=True,
                                   label="Results (click a column header to sort)")
            csv_file = gr.File(label="Download CSV")
            csv_btn.click(predict_csv, inputs=[csv_in, version_dd],
                          outputs=[csv_tbl, csv_file])

        gr.Markdown(
            f"Verdict threshold = {THRESHOLD}. P(detectable) \u2265 {THRESHOLD} \u2192 "
            "*likely detectable*; below \u2192 *likely OVERLOOKED*.\n\n"
            "Inspired by Saeed Lab (FIU) research on recovering overlooked peptides "
            "in MS-based omics. (The ESM-2 model, if selected, loads on first use and "
            "is heavier than the baseline/CNN.)"
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
