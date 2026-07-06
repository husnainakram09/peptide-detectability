"""Evaluate any saved model version on the held-out test split.

Reads a version directory's ``config.json`` to learn the model type
(baseline / cnn / esm2), rebuilds the right features / encoding, scores
``data/processed/test.csv``, and computes accuracy, precision, recall, f1,
roc_auc and pr_auc.

CLI:
    python -m src.evaluate --version v002_cnn_20250101_120000

``score_sequences(version, sequences)`` is reusable for arbitrary inputs (used
by the benchmark and, later, the Gradio app / ``src.predict``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, versioning
from src.train import compute_metrics


# --------------------------------------------------------------------------- #
# Locating a version + its config
# --------------------------------------------------------------------------- #
def resolve_version_dir(version) -> Path:
    """Accept a full directory name, a path, or a bare ``vNNN_*`` name."""
    p = Path(version)
    if p.exists() and p.is_dir():
        return p
    candidate = config.MODELS_DIR / str(version)
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"Version directory not found: {version!r} (looked under {config.MODELS_DIR})"
    )


def load_config(version_dir: Path) -> dict:
    cfg_path = version_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} is missing — is {version_dir.name} a valid model dir?")
    return json.loads(cfg_path.read_text())


# --------------------------------------------------------------------------- #
# Parameter counting (for the benchmark table)
# --------------------------------------------------------------------------- #
def _torch_param_count(state_dict) -> int:
    return int(sum(int(v.numel()) for v in state_dict.values()))


def _lightgbm_param_count(model) -> int:
    """Total number of leaves across all trees (a fair 'param count' for a GBM)."""
    try:
        info = model.booster_.dump_model()["tree_info"]
        return int(sum(t.get("num_leaves", 0) for t in info))
    except Exception:
        try:
            return int(model.booster_.num_trees())
        except Exception:
            return 0


# --------------------------------------------------------------------------- #
# Per-model scoring (each returns (probabilities, param_count))
# --------------------------------------------------------------------------- #
def _torch_infer(model, x, batch_size: int = 256) -> np.ndarray:
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            probs.append(torch.sigmoid(model(x[i:i + batch_size])).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=float)


def _score_baseline(vdir: Path, sequences) -> "tuple[np.ndarray, int]":
    import joblib

    from src import features

    model = joblib.load(vdir / "model.pkl")
    scaler = joblib.load(vdir / "scaler.pkl")
    x = features.featurize_many(sequences)
    probs = model.predict_proba(scaler.transform(x))[:, 1]
    return np.asarray(probs, dtype=float), _lightgbm_param_count(model)


def _score_cnn(vdir: Path, sequences) -> "tuple[np.ndarray, int]":
    import torch

    from src import encoders
    from src.models import cnn as cnn_model

    enc_cfg = json.loads((vdir / "encoder_config.json").read_text())
    arch = dict(enc_cfg["model"])
    model = cnn_model.PeptideCNN(
        vocab_size=enc_cfg["vocab_size"], pad_idx=enc_cfg["pad_idx"], **arch
    )
    state = torch.load(vdir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    x = torch.from_numpy(
        encoders.encode_sequences(sequences, max_len=enc_cfg["max_len"])
    ).long()
    return _torch_infer(model, x), _torch_param_count(state)


def _score_esm2(vdir: Path, sequences, data_hash=None) -> "tuple[np.ndarray, int]":
    import torch

    from src import encoders
    from src.models import esm_head as esm_head_model

    cfg = json.loads((vdir / "config.json").read_text())
    model_id = cfg["esm_model_id"]
    emb_dim = int(cfg["emb_dim"])
    head = dict(cfg.get("head", {"hidden_dim": 128, "dropout": 0.3}))

    if data_hash is not None:
        # reuse the cached test embeddings when scoring the canonical test set
        x_np = encoders.cached_esm_embeddings(sequences, "test", data_hash, model_id=model_id)
    else:
        x_np = encoders.extract_esm_embeddings(sequences, model_id=model_id)

    model = esm_head_model.ESMHead(emb_dim=emb_dim, **head)
    state = torch.load(vdir / "mlp_head.pt", map_location="cpu")
    model.load_state_dict(state)
    return _torch_infer(model, torch.tensor(x_np, dtype=torch.float32)), _torch_param_count(state)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def score_sequences(version, sequences) -> np.ndarray:
    """Return P(detectable) for arbitrary sequences using a saved version."""
    vdir = resolve_version_dir(version)
    model_type = load_config(vdir)["model_type"]
    if model_type == "baseline":
        probs, _ = _score_baseline(vdir, sequences)
    elif model_type == "cnn":
        probs, _ = _score_cnn(vdir, sequences)
    elif model_type == "esm2":
        probs, _ = _score_esm2(vdir, sequences, data_hash=None)
    else:
        raise SystemExit(f"Unknown model_type {model_type!r} in {vdir.name}")
    return probs


def evaluate_on_test(version) -> dict:
    """Score ``data/processed/test.csv`` with ``version`` and return a result dict:
    ``{version, model_type, param_count, y_true, y_prob, metrics}``.
    """
    if not Path(config.TEST_CSV).exists():
        raise SystemExit(
            f"Missing {config.TEST_CSV}. Run `python -m src.data_prep` first."
        )
    vdir = resolve_version_dir(version)
    model_type = load_config(vdir)["model_type"]
    test = pd.read_csv(config.TEST_CSV)
    y_true = test["label"].to_numpy().astype(int)
    sequences = test["sequence"].tolist()
    test_hash = versioning.data_hash_of(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)

    if model_type == "baseline":
        y_prob, n_params = _score_baseline(vdir, sequences)
    elif model_type == "cnn":
        y_prob, n_params = _score_cnn(vdir, sequences)
    elif model_type == "esm2":
        y_prob, n_params = _score_esm2(vdir, sequences, test_hash)
    else:
        raise SystemExit(f"Unknown model_type {model_type!r} in {vdir.name}")

    return {
        "version": vdir.name,
        "model_type": model_type,
        "param_count": int(n_params),
        "y_true": y_true,
        "y_prob": np.asarray(y_prob, dtype=float),
        "metrics": compute_metrics(y_true, y_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model version on the test split.")
    parser.add_argument("--version", required=True,
                        help="Version directory name (e.g. v002_cnn_20250101_120000) or path.")
    args = parser.parse_args()

    result = evaluate_on_test(args.version)
    m = result["metrics"]
    print(f"Version : {result['version']}")
    print(f"Type    : {result['model_type']}   |   param_count: {result['param_count']:,}")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"  {key:<10}: {m[key]:.4f}")
    cm = m["confusion_matrix"]
    print(f"  confusion : TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")


if __name__ == "__main__":
    main()
