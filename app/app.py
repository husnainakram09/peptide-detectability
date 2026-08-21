"""Gradio demo for the peptide detectability predictor.

Tabs:
    1. Single peptide  -> P(detectable) + verdict
    2. Batch (paste)   -> sortable results table + Download CSV
    3. CSV upload      -> same table + Download CSV

A dropdown selects the model version from models/registry.json,
defaulting to the champion model.

The app performs inference only and is CPU-friendly.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make the repository root importable when running:
# python app/app.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Gradio 4.44.x / gradio_client 1.3.0 JSON-schema compatibility fix.
try:
    import gradio_client.utils as _gradio_client_utils

    _original_get_type = _gradio_client_utils.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _original_get_type(schema)

    _gradio_client_utils.get_type = _patched_get_type
except Exception:
    pass

import gradio as gr
import pandas as pd

from src import predict as predictor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

THRESHOLD = 0.5
_COLUMNS = ["sequence", "probability", "verdict"]

# Cache loaded predictors so models are loaded only once.
_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def _versions() -> list:
    """Return available model versions."""
    try:
        return [entry["version"] for entry in predictor.list_versions()]
    except Exception:
        return []


def _default_version():
    """Return champion model if available, otherwise the latest version."""
    try:
        champion = predictor.champion_version()
    except Exception:
        champion = None

    versions = _versions()

    if champion and champion in versions:
        return champion

    return versions[-1] if versions else None


def _entries() -> dict:
    """Return model registry entries indexed by version."""
    try:
        return {
            entry["version"]: entry
            for entry in predictor.list_versions()
        }
    except Exception:
        return {}


def _model_info_md(version) -> str:
    """Create Markdown describing the active model."""
    if not version:
        return (
            "**No model available.** "
            "Train one first, e.g. "
            "`python -m src.train --model baseline`."
        )

    entry = _entries().get(version, {})
    champion = predictor.champion_version()

    star = " — ★ champion" if version == champion else ""

    metrics = entry.get("metrics", {})
    bits = []

    for key in ("pr_auc", "roc_auc", "f1"):
        value = metrics.get(key)

        if isinstance(value, (int, float)):
            bits.append(f"{key}={value:.3f}")

    metric_str = (
        "  |  " + ", ".join(bits)
        if bits
        else ""
    )

    return (
        f"**Active model:** `{version}` "
        f"(type: {entry.get('model_type', '?')})"
        f"{star}{metric_str}"
    )


def _get_predictor(version):
    """Load and cache the selected model."""
    if version not in _CACHE:
        _CACHE[version] = predictor.load_model(version)

    return _CACHE[version]


# --------------------------------------------------------------------------- #
# Input / prediction helpers
# --------------------------------------------------------------------------- #

def _clean_seq(sequence) -> str:
    """Normalize a peptide sequence."""
    return "".join(
        character
        for character in str(sequence).strip().upper()
        if character.isalpha()
    )


def _verdict(probability: float) -> str:
    """Return human-readable prediction verdict."""
    if probability >= THRESHOLD:
        return "likely detectable"

    return "likely OVERLOOKED"


def _empty_df() -> pd.DataFrame:
    """Return an empty prediction DataFrame."""
    return pd.DataFrame(columns=_COLUMNS)


def _predict_frame(sequences, version):
    """Predict a list of sequences and create a downloadable CSV."""
    if not version:
        return _empty_df(), None

    cleaned_sequences = [
        _clean_seq(sequence)
        for sequence in sequences
    ]

    cleaned_sequences = [
        sequence
        for sequence in cleaned_sequences
        if sequence
    ]

    if not cleaned_sequences:
        return _empty_df(), None

    probabilities = _get_predictor(version)(cleaned_sequences)

    df = pd.DataFrame(
        {
            "sequence": cleaned_sequences,
            "probability": [
                round(float(probability), 4)
                for probability in probabilities
            ],
            "verdict": [
                _verdict(float(probability))
                for probability in probabilities
            ],
        }
    )

    tmp = tempfile.NamedTemporaryFile(
        prefix="peptide_predictions_",
        suffix=".csv",
        delete=False,
    )
    tmp.close()

    df.to_csv(tmp.name, index=False)

    return df, tmp.name


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

def predict_single(sequence, version):
    """Predict detectability for a single peptide."""
    if not version:
        return "No model available — train one first."

    sequence = _clean_seq(sequence)

    if not sequence:
        return "Enter a peptide sequence (letters only)."

    probability = float(
        _get_predictor(version)([sequence])[0]
    )

    mark = "✅" if probability >= THRESHOLD else "🚫"

    return (
        f"### {mark} {_verdict(probability)}\n\n"
        f"**P(detectable) = {probability:.3f}** "
        f"for `{sequence}` "
        f"(length {len(sequence)})"
    )


def predict_batch(text, version):
    """Predict peptides entered one per line."""
    lines = (text or "").splitlines()

    return _predict_frame(lines, version)


def predict_csv(file, version):
    """Predict peptides from an uploaded CSV file."""
    if file is None:
        return _empty_df(), None

    path = (
        file
        if isinstance(file, str)
        else getattr(file, "name", None)
    )

    if not path:
        return _empty_df(), None

    df_input = pd.read_csv(path)

    sequence_column = next(
        (
            column
            for column in df_input.columns
            if str(column).strip().lower() == "sequence"
        ),
        None,
    )

    if sequence_column is None:
        sequence_column = df_input.columns[0]

    return _predict_frame(
        df_input[sequence_column].tolist(),
        version,
    )


def refresh_models():
    """Refresh available models and update the model information."""
    versions = _versions()
    default = _default_version()

    # Return a new Dropdown component rather than gr.update().
    updated_dropdown = gr.Dropdown(
        choices=versions,
        value=default,
        label="Model version",
        scale=4,
    )

    return updated_dropdown, _model_info_md(default)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def build_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    versions = _versions()
    default = _default_version()

    with gr.Blocks(
        title="Peptide Detectability Predictor"
    ) as demo:

        gr.Markdown(
            "# 🧬 Peptide Detectability Predictor\n"
            "Predicts whether a peptide is likely to be **detected** "
            "(a proteotypic “flyer”) in a bottom-up mass-spectrometry "
            "experiment, or likely **overlooked**. "
            "Inference only — CPU-friendly."
        )

        # ------------------------------------------------------------------ #
        # Model selection
        # ------------------------------------------------------------------ #

        with gr.Row():

            version_dd = gr.Dropdown(
                choices=versions,
                value=default,
                label="Model version",
                scale=4,
            )

            refresh_btn = gr.Button(
                "🔄 Refresh models",
                scale=1,
            )

        active_md = gr.Markdown(
            _model_info_md(default)
        )

        version_dd.change(
            fn=_model_info_md,
            inputs=version_dd,
            outputs=active_md,
        )

        refresh_btn.click(
            fn=refresh_models,
            outputs=[version_dd, active_md],
        )

        # ------------------------------------------------------------------ #
        # Single peptide
        # ------------------------------------------------------------------ #

        with gr.Tab("Single peptide"):

            seq_in = gr.Textbox(
                label="Peptide sequence",
                placeholder="e.g. TITLEVEPSDTIENVK",
            )

            single_btn = gr.Button(
                "Predict",
                variant="primary",
            )

            single_out = gr.Markdown()

            single_btn.click(
                fn=predict_single,
                inputs=[seq_in, version_dd],
                outputs=single_out,
            )

            seq_in.submit(
                fn=predict_single,
                inputs=[seq_in, version_dd],
                outputs=single_out,
            )

        # ------------------------------------------------------------------ #
        # Batch prediction
        # ------------------------------------------------------------------ #

        with gr.Tab("Batch (paste)"):

            multi_in = gr.Textbox(
                label="One peptide per line",
                lines=8,
                placeholder=(
                    "TITLEVEPSDTIENVK\n"
                    "HQGVMVGMGQK\n"
                    "ESTLHLVLR"
                ),
            )

            batch_btn = gr.Button(
                "Predict batch",
                variant="primary",
            )

            batch_tbl = gr.Dataframe(
                headers=_COLUMNS,
                interactive=False,
                wrap=True,
                label="Results (click a column header to sort)",
            )

            batch_file = gr.File(
                label="Download CSV"
            )

            batch_btn.click(
                fn=predict_batch,
                inputs=[multi_in, version_dd],
                outputs=[batch_tbl, batch_file],
            )

        # ------------------------------------------------------------------ #
        # CSV prediction
        # ------------------------------------------------------------------ #

        with gr.Tab("CSV upload"):

            csv_in = gr.File(
                label="CSV with a 'sequence' column",
                file_types=[".csv"],
                type="filepath",
            )

            csv_btn = gr.Button(
                "Predict from CSV",
                variant="primary",
            )

            csv_tbl = gr.Dataframe(
                headers=_COLUMNS,
                interactive=False,
                wrap=True,
                label="Results (click a column header to sort)",
            )

            csv_file = gr.File(
                label="Download CSV"
            )

            csv_btn.click(
                fn=predict_csv,
                inputs=[csv_in, version_dd],
                outputs=[csv_tbl, csv_file],
            )

        # ------------------------------------------------------------------ #
        # Footer
        # ------------------------------------------------------------------ #

        gr.Markdown(
            f"Verdict threshold = {THRESHOLD}. "
            f"P(detectable) ≥ {THRESHOLD} → "
            "*likely detectable*; below → "
            "*likely OVERLOOKED*."
        )

    return demo


# --------------------------------------------------------------------------- #
# Local execution
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    build_app().launch()