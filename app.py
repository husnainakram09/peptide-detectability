"""Gradio demo for the peptide detectability predictor (Hugging Face Space entrypoint).

Tabs:
    1. Single peptide  -> P(detectable) + verdict
    2. Batch (paste)   -> sortable results table + Download CSV
    3. CSV upload      -> same table + Download CSV

A dropdown selects the model version from models/registry.json, defaulting to
the champion. Only peptides inside the model's training domain (20 standard
amino acids, length MIN_LEN..MAX_LEN) are scored; anything else is reported as
"not scored" instead of receiving a meaningless probability.

Inference only, CPU-friendly. Run locally with:  python app.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from src import config
from src import predict as predictor
from src import validation
from src.validation import MAX_ROWS, THRESHOLD


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MAX_UPLOAD = "5mb"
_COLUMNS = ["sequence", "length", "probability", "verdict"]

# Downloadable CSVs are written here and pruned after an hour.
_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "peptide_predictions"
_DOWNLOAD_TTL_S = 3600

# Cache loaded predictors so models are loaded only once.
_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def _entries() -> dict:
    """Return model registry entries indexed by version."""
    try:
        return {entry["version"]: entry for entry in predictor.list_versions()}
    except Exception:
        return {}


def _versions() -> list:
    """Return available model versions."""
    return list(_entries())


def _default_version():
    """Return champion model if available, otherwise the latest version."""
    champion = predictor.champion_version()
    versions = _versions()
    if champion and champion in versions:
        return champion
    return versions[-1] if versions else None


def _model_info_md(version) -> str:
    """Create Markdown describing the active model."""
    if not version:
        return (
            "**No model available.** "
            "Train one first, e.g. `python -m src.train --model baseline`."
        )

    entry = _entries().get(version, {})
    star = " — ★ champion" if version == predictor.champion_version() else ""

    metrics = entry.get("metrics", {})
    bits = [
        f"{key}={metrics[key]:.3f}"
        for key in ("pr_auc", "roc_auc", "f1")
        if isinstance(metrics.get(key), (int, float))
    ]
    metric_str = "  |  test " + ", ".join(bits) if bits else ""

    return (
        f"**Active model:** `{version}` "
        f"(type: {entry.get('model_type', '?')}){star}{metric_str}"
    )


def _get_predictor(version):
    """Load and cache the selected model."""
    if version not in _CACHE:
        _CACHE[version] = predictor.load_model(version)
    return _CACHE[version]


# --------------------------------------------------------------------------- #
# Input / prediction helpers
# --------------------------------------------------------------------------- #

def _empty_df() -> pd.DataFrame:
    """Return an empty prediction DataFrame."""
    return pd.DataFrame(columns=_COLUMNS)


def _write_csv(df: pd.DataFrame) -> str:
    """Write ``df`` to a downloadable CSV, pruning files older than the TTL."""
    _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - _DOWNLOAD_TTL_S
    for old in _DOWNLOAD_DIR.glob("*.csv"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass

    tmp = tempfile.NamedTemporaryFile(
        prefix="peptide_predictions_",
        suffix=".csv",
        dir=_DOWNLOAD_DIR,
        delete=False,
    )
    tmp.close()
    df.to_csv(tmp.name, index=False)
    return tmp.name


def _predict_frame(sequences, version):
    """Score a list of sequences.

    Returns ``(results DataFrame, CSV path or None, status Markdown)``.
    Out-of-domain sequences are kept in the table with an empty probability.
    """
    if not version:
        return _empty_df(), None, "No model available."

    try:
        rows = validation.score_sequences(sequences, _get_predictor(version))
    except validation.TooManySequences as exc:
        return _empty_df(), None, f"⚠️ {exc}"

    if not rows:
        return _empty_df(), None, "No sequences provided."

    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df, _write_csv(df), validation.summary(rows)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

def predict_single(sequence, version):
    """Predict detectability for a single peptide."""
    if not version:
        return "No model available — train one first."

    sequence = validation.clean_sequence(sequence)

    if not sequence:
        return "Enter a peptide sequence."

    problem = validation.validate_sequence(sequence)
    if problem:
        shown = sequence if len(sequence) <= 60 else sequence[:60] + "…"
        return (
            f"### ⚠️ Not scored\n\n`{shown}`: {problem}.\n\n"
            f"{validation.domain_note()}"
        )

    probability = float(_get_predictor(version)([sequence])[0])
    mark = "✅" if probability >= THRESHOLD else "🚫"

    return (
        f"### {mark} {validation.verdict(probability)}\n\n"
        f"**P(detectable) = {probability:.3f}** "
        f"for `{sequence}` (length {len(sequence)})"
    )


def predict_batch(text, version):
    """Predict peptides entered one per line."""
    return _predict_frame((text or "").splitlines(), version)


def predict_csv(file, version):
    """Predict peptides from an uploaded CSV file."""
    if file is None:
        return _empty_df(), None, "Upload a CSV file."

    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path:
        return _empty_df(), None, "Upload a CSV file."

    try:
        # Read one row past the limit so _predict_frame can report the overflow
        # without loading an arbitrarily large file.
        df_input = pd.read_csv(path, dtype=str, nrows=MAX_ROWS + 1)
    except Exception as exc:
        return _empty_df(), None, f"⚠️ Could not read the CSV: {exc}"

    if df_input.empty:
        return _empty_df(), None, "The CSV has no rows."

    sequence_column = next(
        (
            column
            for column in df_input.columns
            if str(column).strip().lower() == "sequence"
        ),
        df_input.columns[0],
    )

    return _predict_frame(
        df_input[sequence_column].dropna().tolist(),
        version,
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def build_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    versions = _versions()
    default = _default_version()

    with gr.Blocks(
        title="Peptide Detectability Predictor",
        delete_cache=(_DOWNLOAD_TTL_S, _DOWNLOAD_TTL_S),
    ) as demo:

        gr.Markdown(
            "# 🧬 Peptide Detectability Predictor\n"
            "Predicts whether a tryptic peptide is likely to be **detected** "
            "(a proteotypic “flyer”) in a bottom-up mass-spectrometry "
            "experiment, or likely **overlooked**. Trained on GPMDB "
            "observation data; for research use only.\n\n"
            f"Input: unmodified peptides of {config.MIN_LEN}–{config.MAX_LEN} "
            "residues over the 20 standard amino acids. Up to "
            f"{MAX_ROWS:,} sequences per batch or CSV."
        )

        # ------------------------------------------------------------------ #
        # Model selection
        # ------------------------------------------------------------------ #

        version_dd = gr.Dropdown(
            choices=versions,
            value=default,
            label="Model version",
        )

        active_md = gr.Markdown(_model_info_md(default))

        version_dd.change(
            fn=_model_info_md,
            inputs=version_dd,
            outputs=active_md,
        )

        # ------------------------------------------------------------------ #
        # Single peptide
        # ------------------------------------------------------------------ #

        with gr.Tab("Single peptide"):

            seq_in = gr.Textbox(
                label="Peptide sequence",
                placeholder="e.g. TITLEVEPSDTIENVK",
            )

            single_btn = gr.Button("Predict", variant="primary")

            single_out = gr.Markdown()

            gr.Examples(
                examples=[["TITLEVEPSDTIENVK"], ["HQGVMVGMGQK"], ["ESTLHLVLR"]],
                inputs=seq_in,
            )

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
                label=f"One peptide per line (max {MAX_ROWS:,})",
                lines=8,
                placeholder="TITLEVEPSDTIENVK\nHQGVMVGMGQK\nESTLHLVLR",
            )

            batch_btn = gr.Button("Predict batch", variant="primary")

            batch_status = gr.Markdown()

            batch_tbl = gr.Dataframe(
                headers=_COLUMNS,
                interactive=False,
                wrap=True,
                label="Results (click a column header to sort)",
            )

            batch_file = gr.File(label="Download CSV")

            batch_btn.click(
                fn=predict_batch,
                inputs=[multi_in, version_dd],
                outputs=[batch_tbl, batch_file, batch_status],
            )

        # ------------------------------------------------------------------ #
        # CSV prediction
        # ------------------------------------------------------------------ #

        with gr.Tab("CSV upload"):

            csv_in = gr.File(
                label=(
                    "CSV with a 'sequence' column (otherwise the first column "
                    f"is used; max {MAX_ROWS:,} rows, {MAX_UPLOAD})"
                ),
                file_types=[".csv"],
                type="filepath",
            )

            csv_btn = gr.Button("Predict from CSV", variant="primary")

            csv_status = gr.Markdown()

            csv_tbl = gr.Dataframe(
                headers=_COLUMNS,
                interactive=False,
                wrap=True,
                label="Results (click a column header to sort)",
            )

            csv_file = gr.File(label="Download CSV")

            csv_btn.click(
                fn=predict_csv,
                inputs=[csv_in, version_dd],
                outputs=[csv_tbl, csv_file, csv_status],
            )

        # ------------------------------------------------------------------ #
        # Footer
        # ------------------------------------------------------------------ #

        gr.Markdown(
            f"Verdict threshold = {THRESHOLD}. "
            f"P(detectable) ≥ {THRESHOLD} → *likely detectable*; "
            "below → *likely OVERLOOKED*. "
            "Data: DeepMSPeptide / GPMDB (Serrano et al., *Bioinformatics* "
            "2020). Predictions are a screening aid, not a guarantee of "
            "detection on any particular instrument."
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    # On Hugging Face Spaces, GRADIO_SERVER_NAME / GRADIO_SERVER_PORT are set
    # by the platform; locally this binds to 127.0.0.1:7860.
    demo.launch(max_file_size=MAX_UPLOAD)
