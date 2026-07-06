"""LightGBM baseline model for peptide detectability.

A gradient-boosted tree classifier trained on the engineered features from
``src.features`` (composition + physicochemical descriptors). Uses class
weighting (``scale_pos_weight``) for imbalance and early stopping on the
validation set. Fast on CPU.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np


def compute_scale_pos_weight(y) -> float:
    """neg / pos ratio for LightGBM's ``scale_pos_weight`` (1.0 if no positives)."""
    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def default_params(seed: int, scale_pos_weight: "float | None" = None) -> "dict[str, Any]":
    """Reproducible, CPU-friendly LightGBM hyperparameters."""
    params: dict[str, Any] = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    # Class weighting for imbalance: prefer an explicit scale_pos_weight (recorded
    # in the run config); fall back to is_unbalance if none is supplied.
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    else:
        params["is_unbalance"] = True
    return params


def train_baseline(X_train, y_train, X_val, y_val, params: "dict[str, Any]",
                   early_stopping_rounds: int = 50) -> "lgb.LGBMClassifier":
    """Fit a LightGBM classifier with early stopping on the validation set.

    Early stopping monitors average precision (PR-AUC) on the val split — the
    project's primary metric.
    """
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                               first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model
