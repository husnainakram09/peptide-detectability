"""MLP head over frozen ESM-2 embeddings.

A tiny classifier on top of mean-pooled ESM-2 embeddings::

    Linear(emb_dim -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)

Outputs a single logit per peptide; train with ``BCEWithLogitsLoss(pos_weight=...)``.
``emb_dim`` is 320 for ``esm2_t6_8M`` and 480 for ``esm2_t12_35M``. Because the
ESM-2 backbone is frozen (embeddings are cached), this head trains in seconds on
CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ESMHead(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, emb_dim) -> logit (B,)
        return self.net(x).squeeze(1)


def pos_weight_from_labels(y) -> float:
    """neg / pos ratio, for ``BCEWithLogitsLoss(pos_weight=...)`` (1.0 if no positives)."""
    import numpy as np

    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0
