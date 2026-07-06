"""Load a saved model version (or the champion) and predict on new peptides.

Self-contained variant for the Hugging Face Space: identical to the repo's
``src/predict.py`` but with ``resolve_version_dir`` and ``load_config`` inlined,
so the Space needs only these modules (no ``evaluate.py`` / ``train.py``).

``load_model(version_dir)`` reads the version's ``config.json`` and returns a
callable ``predict(sequences) -> np.ndarray`` of P(detectable), for all three
model types (baseline / cnn / esm2). Artifacts load once and are reused.
Inference only; CPU-only by default.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config, versioning


# --------------------------------------------------------------------------- #
# Locating a version + its config (inlined so the Space is self-contained)
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
# Registry helpers
# --------------------------------------------------------------------------- #
def list_versions() -> "list[dict]":
    """All registered runs (oldest first), from ``models/registry.json``."""
    return versioning.list_versions()


def champion_version() -> "str | None":
    """The champion version name from ``models/LATEST.json`` (or None)."""
    try:
        return json.loads(versioning.LATEST.read_text())["champion"]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Per-model loaders — each returns a bound predict(sequences) -> np.ndarray
# --------------------------------------------------------------------------- #
def _load_baseline(vdir: Path):
    import joblib

    from src import features

    model = joblib.load(vdir / "model.pkl")
    scaler = joblib.load(vdir / "scaler.pkl")

    def predict(sequences) -> np.ndarray:
        seqs = list(sequences)
        if not seqs:
            return np.zeros(0, dtype=float)
        x = features.featurize_many(seqs)
        return model.predict_proba(scaler.transform(x))[:, 1].astype(float)

    return predict


def _load_cnn(vdir: Path):
    import torch

    from src import encoders
    from src.models import cnn as cnn_model

    enc_cfg = json.loads((vdir / "encoder_config.json").read_text())
    arch = dict(enc_cfg["model"])
    max_len = int(enc_cfg["max_len"])
    model = cnn_model.PeptideCNN(
        vocab_size=enc_cfg["vocab_size"], pad_idx=enc_cfg["pad_idx"], **arch
    )
    model.load_state_dict(torch.load(vdir / "model.pt", map_location="cpu"))
    model.eval()

    def predict(sequences) -> np.ndarray:
        seqs = list(sequences)
        if not seqs:
            return np.zeros(0, dtype=float)
        x = torch.from_numpy(encoders.encode_sequences(seqs, max_len=max_len)).long()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                out.append(torch.sigmoid(model(x[i:i + 256])).cpu().numpy())
        return np.concatenate(out).astype(float)

    return predict


def _load_esm2(vdir: Path):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from src.models import esm_head as esm_head_model

    cfg = json.loads((vdir / "config.json").read_text())
    model_id = cfg["esm_model_id"]
    emb_dim = int(cfg["emb_dim"])
    head = dict(cfg.get("head", {"hidden_dim": 128, "dropout": 0.3}))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    esm = AutoModel.from_pretrained(model_id).to(device).eval()          # loaded once
    head_model = esm_head_model.ESMHead(emb_dim=emb_dim, **head)
    head_model.load_state_dict(torch.load(vdir / "mlp_head.pt", map_location="cpu"))
    head_model.eval()
    max_len = config.MAX_LEN + 2  # room for BOS/EOS

    def predict(sequences) -> np.ndarray:
        seqs = [str(s).strip().upper() for s in sequences]
        if not seqs:
            return np.zeros(0, dtype=float)
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                batch = seqs[i:i + 16]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_len, return_special_tokens_mask=True)
                special = enc.pop("special_tokens_mask").to(device)
                enc = {k: v.to(device) for k, v in enc.items()}
                hidden = esm(**enc).last_hidden_state
                residue = (enc["attention_mask"] * (1 - special)).unsqueeze(-1).float()
                emb = (hidden * residue).sum(1) / residue.sum(1).clamp(min=1.0)
                out.append(torch.sigmoid(head_model(emb)).cpu().numpy())
        return np.concatenate(out).astype(float)

    return predict


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_model(version_dir):
    """Return a callable ``predict(sequences) -> np.ndarray`` for a saved version."""
    vdir = resolve_version_dir(version_dir)
    model_type = load_config(vdir)["model_type"]
    if model_type == "baseline":
        return _load_baseline(vdir)
    if model_type == "cnn":
        return _load_cnn(vdir)
    if model_type == "esm2":
        return _load_esm2(vdir)
    raise ValueError(f"Unknown model_type {model_type!r} in {vdir.name}")


def load_champion():
    """Load the current champion model. Returns ``(version_name, predict_fn)``."""
    champ = champion_version()
    if champ is None:
        raise SystemExit("No champion found. Copy trained models into space/models/ first.")
    return champ, load_model(champ)
