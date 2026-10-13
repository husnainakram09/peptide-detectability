"""Training entrypoint.

Usage:
    python -m src.train --model {baseline,cnn,esm2} [--esm-model MODEL_ID]

Every run writes a brand-new, timestamped version directory under ``models/``
(via ``src.versioning``) and updates the champion pointer in ``models/LATEST.json``.
No run ever overwrites a previous model.

Implemented: ``baseline`` (Phase 3), ``cnn`` (Phase 4), ``esm2`` (Phase 5).
Model-specific heavy dependencies (LightGBM/Biopython for baseline, PyTorch for
cnn, PyTorch+transformers for esm2) are imported lazily inside each ``run_*``
function so that training one model never requires the other's libraries.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config, versioning

# ESM-2 model options (plain strings so argparse needs no heavy imports).
_ESM_MODELS = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_splits() -> "tuple":
    import pandas as pd

    for p in (config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV):
        if not Path(p).exists():
            raise SystemExit(
                f"Missing {p}. Run `python -m src.data_prep` first to create the "
                f"processed train/val/test splits."
            )
    return (
        pd.read_csv(config.TRAIN_CSV),
        pd.read_csv(config.VAL_CSV),
        pd.read_csv(config.TEST_CSV),
    )


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """accuracy / precision / recall / f1 / roc_auc / pr_auc + confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "threshold": float(threshold),
    }


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} — TEST metrics ===")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"  {key:<10}: {metrics[key]:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  confusion : TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")


def _announce_save(vdir: Path, champ: dict) -> None:
    print(f"\n[save] model directory : {vdir.relative_to(config.PROJECT_ROOT)}")
    print(f"[save] registry updated : models/registry.json")
    print(f"[save] champion is now   : {champ['champion']} "
          f"({champ['metric']}={champ['value']:.4f})")


# --------------------------------------------------------------------------- #
# Shared PyTorch training loop (used by cnn and esm2)
# --------------------------------------------------------------------------- #
def _torch_predict_proba(model, x, device, batch_size: int = 256) -> np.ndarray:
    """Run a torch model in eval mode and return P(detectable) for each row."""
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=float)


def _train_torch_head(model, x_train, y_train, x_val, y_val, *, seed: int,
                      lr: float, batch_size: int, max_epochs: int, patience: int,
                      pos_weight: float, device, tag: str) -> "tuple[float, int]":
    """Train ``model`` (already on ``device``) with Adam + early stopping on val
    PR-AUC. Restores the best-epoch weights in place. Returns (best_pr, best_epoch).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size,
        shuffle=True, generator=generator,
    )

    y_val_np = y_val.numpy().astype(int)
    best_pr, best_epoch, best_state, waited = -1.0, -1, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss) * xb.size(0)
        val_prob = _torch_predict_proba(model, x_val, device, batch_size)
        val_pr = float(average_precision_score(y_val_np, val_prob))
        print(f"  [{tag}] epoch {epoch:2d}/{max_epochs}  "
              f"train_loss={running / len(x_train):.4f}  val_pr_auc={val_pr:.4f}")
        if val_pr > best_pr:
            best_pr, best_epoch, waited = val_pr, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  [{tag}] early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val_pr_auc={best_pr:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_pr, best_epoch


# --------------------------------------------------------------------------- #
# Baseline (Phase 3)
# --------------------------------------------------------------------------- #
def run_baseline(seed: int) -> None:
    import joblib
    from sklearn.preprocessing import StandardScaler

    from src import features
    from src.models import baseline as baseline_model

    print("[baseline] loading processed splits ...")
    train, val, test = _load_splits()

    print("[baseline] building features (20-AA composition + physicochemical) ...")
    x_train = features.featurize_dataframe(train, progress=True)
    x_val = features.featurize_dataframe(val, progress=True)
    x_test = features.featurize_dataframe(test, progress=True)
    y_train = train["label"].to_numpy().astype(int)
    y_val = val["label"].to_numpy().astype(int)
    y_test = test["label"].to_numpy().astype(int)

    print("[baseline] fitting StandardScaler on TRAIN only ...")
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    spw = baseline_model.compute_scale_pos_weight(y_train)
    params = baseline_model.default_params(seed=seed, scale_pos_weight=spw)
    print(f"[baseline] training LightGBM (scale_pos_weight={spw:.3f}), "
          f"early stopping on val PR-AUC ...")
    model = baseline_model.train_baseline(x_train_s, y_train, x_val_s, y_val, params)
    best_iter = getattr(model, "best_iteration_", None)

    print("[baseline] evaluating on TEST ...")
    y_test_prob = model.predict_proba(x_test_s)[:, 1]
    metrics = compute_metrics(y_test, y_test_prob)
    _print_metrics("LightGBM baseline", metrics)

    data_hash = versioning.data_hash_of(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)
    vdir = versioning.new_version_dir("baseline")
    joblib.dump(model, vdir / "model.pkl")
    joblib.dump(scaler, vdir / "scaler.pkl")

    cfg = {
        "model_type": "baseline",
        "seed": seed,
        "feature_names": features.FEATURE_NAMES,
        "n_features": features.N_FEATURES,
        "scaler": "StandardScaler (fit on train only)",
        "lightgbm_params": params,
        "best_iteration": int(best_iter) if best_iter else None,
        "primary_metric": config.PRIMARY_METRIC,
        "data_hash": data_hash,
        "artifacts": {"model": "model.pkl", "scaler": "scaler.pkl"},
    }
    versioning.register(vdir, "baseline", metrics, cfg, data_hash)
    champ = versioning.update_champion(metric=config.PRIMARY_METRIC)
    _announce_save(vdir, champ)


# --------------------------------------------------------------------------- #
# 1D-CNN (Phase 4)
# --------------------------------------------------------------------------- #
def run_cnn(seed: int) -> None:
    import torch

    from src import encoders
    from src.models import cnn as cnn_model

    torch.manual_seed(seed)  # numpy + python seeds already set in main()

    print("[cnn] loading processed splits ...")
    train, val, test = _load_splits()

    print(f"[cnn] integer-encoding sequences (pad idx 0, max_len {encoders.MAX_LEN}) ...")
    x_train, y_train = encoders.encode_dataframe(train)
    x_val, y_val = encoders.encode_dataframe(val)
    x_test, y_test = encoders.encode_dataframe(test)

    device = torch.device("cpu")
    lr, batch_size, max_epochs, patience = 1e-3, 64, 30, 5
    model_arch = {
        "embed_dim": 32, "num_channels": 64, "kernel_sizes": [3, 5],
        "dense_dim": 64, "dropout": 0.3,
    }
    model = cnn_model.PeptideCNN(
        vocab_size=encoders.VOCAB_SIZE, pad_idx=encoders.PAD_IDX, **model_arch
    ).to(device)
    pos_weight = cnn_model.pos_weight_from_labels(y_train.numpy())

    print(f"[cnn] training on CPU: Adam(lr={lr}), batch={batch_size}, "
          f"<= {max_epochs} epochs, early stop on val PR-AUC (patience {patience}), "
          f"pos_weight={pos_weight:.3f} ...")
    best_pr, best_epoch = _train_torch_head(
        model, x_train, y_train, x_val, y_val, seed=seed, lr=lr, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, pos_weight=pos_weight,
        device=device, tag="cnn",
    )

    print("[cnn] evaluating on TEST ...")
    test_prob = _torch_predict_proba(model, x_test, device, batch_size)
    metrics = compute_metrics(y_test.numpy().astype(int), test_prob)
    _print_metrics("1D-CNN", metrics)

    data_hash = versioning.data_hash_of(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)
    vdir = versioning.new_version_dir("cnn")
    torch.save(model.state_dict(), vdir / "model.pt")
    # NumPy copy of the weights so the deployed apps can serve without PyTorch.
    from src.models import cnn_numpy
    cnn_numpy.export_weights(model.state_dict(), vdir / "model.npz")

    encoder_config = {
        "vocab": config.AMINO_ACIDS,
        "aa_to_idx": config.AA_TO_IDX,
        "pad_idx": encoders.PAD_IDX,
        "max_len": encoders.MAX_LEN,
        "vocab_size": encoders.VOCAB_SIZE,
        "model": model_arch,
    }
    (vdir / "encoder_config.json").write_text(json.dumps(encoder_config, indent=2))

    cfg = {
        "model_type": "cnn",
        "seed": seed,
        "framework": "pytorch (CPU)",
        "hyperparameters": {
            "optimizer": "Adam", "lr": lr, "batch_size": batch_size,
            "max_epochs": max_epochs, "patience": patience, "best_epoch": best_epoch,
            "loss": "BCEWithLogitsLoss", "pos_weight": pos_weight,
        },
        "model_arch": model_arch,
        "best_val_pr_auc": best_pr,
        "primary_metric": config.PRIMARY_METRIC,
        "data_hash": data_hash,
        "artifacts": {"model": "model.pt", "numpy_model": "model.npz",
                      "encoder_config": "encoder_config.json"},
    }
    versioning.register(vdir, "cnn", metrics, cfg, data_hash)
    champ = versioning.update_champion(metric=config.PRIMARY_METRIC)
    _announce_save(vdir, champ)


# --------------------------------------------------------------------------- #
# ESM-2 + MLP head (Phase 5)
# --------------------------------------------------------------------------- #
def run_esm2(seed: int, model_id: str) -> None:
    import torch

    from src import encoders
    from src.models import esm_head as esm_head_model

    torch.manual_seed(seed)

    print("[esm2] loading processed splits ...")
    train, val, test = _load_splits()

    data_hash = versioning.data_hash_of(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)
    print(f"[esm2] obtaining ESM-2 embeddings ({model_id}) — uses "
          f"embeddings_cache/ if present, else computes (GPU if available) ...")
    x_train_np = encoders.cached_esm_embeddings(train["sequence"].tolist(), "train",
                                                data_hash, model_id=model_id)
    x_val_np = encoders.cached_esm_embeddings(val["sequence"].tolist(), "val",
                                              data_hash, model_id=model_id)
    x_test_np = encoders.cached_esm_embeddings(test["sequence"].tolist(), "test",
                                               data_hash, model_id=model_id)
    emb_dim = int(x_train_np.shape[1])

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_train = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(val["label"].to_numpy(), dtype=torch.float32)
    y_test_np = test["label"].to_numpy().astype(int)

    device = torch.device("cpu")
    lr, batch_size, max_epochs, patience = 1e-3, 64, 50, 5
    head = {"hidden_dim": 128, "dropout": 0.3}
    model = esm_head_model.ESMHead(emb_dim=emb_dim, **head).to(device)
    pos_weight = esm_head_model.pos_weight_from_labels(train["label"].to_numpy())

    print(f"[esm2] training MLP head on {emb_dim}-dim embeddings: Adam(lr={lr}), "
          f"batch={batch_size}, <= {max_epochs} epochs, early stop on val PR-AUC "
          f"(patience {patience}), pos_weight={pos_weight:.3f} ...")
    best_pr, best_epoch = _train_torch_head(
        model, x_train, y_train, x_val, y_val, seed=seed, lr=lr, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, pos_weight=pos_weight,
        device=device, tag="esm2",
    )

    print("[esm2] evaluating on TEST ...")
    test_prob = _torch_predict_proba(model, x_test, device, batch_size)
    metrics = compute_metrics(y_test_np, test_prob)
    _print_metrics(f"ESM-2 + MLP ({encoders.esm_model_tag(model_id)})", metrics)

    data_hash = versioning.data_hash_of(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)
    vdir = versioning.new_version_dir("esm2")
    torch.save(model.state_dict(), vdir / "mlp_head.pt")

    cfg = {
        "model_type": "esm2",
        "seed": seed,
        "framework": "pytorch (CPU); embeddings via HuggingFace transformers",
        "esm_model_id": model_id,
        "esm_model_tag": encoders.esm_model_tag(model_id),
        "emb_dim": emb_dim,
        "head": head,
        "hyperparameters": {
            "optimizer": "Adam", "lr": lr, "batch_size": batch_size,
            "max_epochs": max_epochs, "patience": patience, "best_epoch": best_epoch,
            "loss": "BCEWithLogitsLoss", "pos_weight": pos_weight,
        },
        "best_val_pr_auc": best_pr,
        "primary_metric": config.PRIMARY_METRIC,
        "data_hash": data_hash,
        "artifacts": {"model": "mlp_head.pt"},
    }
    versioning.register(vdir, "esm2", metrics, cfg, data_hash)
    champ = versioning.update_champion(metric=config.PRIMARY_METRIC)
    _announce_save(vdir, champ)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a peptide detectability model.")
    parser.add_argument("--model", required=True, choices=["baseline", "cnn", "esm2"],
                        help="Which model to train.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed (default: config.RANDOM_SEED).")
    parser.add_argument("--esm-model", default=_ESM_MODELS[0], choices=_ESM_MODELS,
                        help="ESM-2 model id (esm2 only). Default: the 8M t6 model.")
    args = parser.parse_args()

    _set_seeds(args.seed)

    if args.model == "baseline":
        run_baseline(args.seed)
    elif args.model == "cnn":
        run_cnn(args.seed)
    elif args.model == "esm2":
        run_esm2(args.seed, args.esm_model)


if __name__ == "__main__":
    main()
