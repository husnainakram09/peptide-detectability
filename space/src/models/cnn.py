"""1D-CNN sequence classifier (DeepMSPeptide-style), PyTorch, CPU-friendly.

Architecture::

    Embedding(vocab_size + 1, 32, padding_idx=0)
      -> two parallel Conv1d branches (kernel sizes 3 and 5, 64 channels, ReLU)
      -> global max-pool over the sequence, per branch
      -> concat  (64 * 2 = 128)
      -> Linear(128 -> 64) -> ReLU -> Dropout(0.3)
      -> Linear(64 -> 1)   [single logit per peptide]

Train with ``BCEWithLogitsLoss(pos_weight=neg/pos)`` for class imbalance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PeptideCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_channels: int = 64,
        kernel_sizes=(3, 5),
        dense_dim: int = 64,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.relu = nn.ReLU()
        concat_dim = num_channels * len(kernel_sizes)
        self.fc1 = nn.Linear(concat_dim, dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) long token ids
        emb = self.embedding(x)             # (B, L, E)
        emb = emb.transpose(1, 2)           # (B, E, L)  -> Conv1d wants (B, C, L)
        pooled = []
        for conv in self.convs:
            h = self.relu(conv(emb))        # (B, C, L)
            h = torch.max(h, dim=2).values  # global max-pool over time -> (B, C)
            pooled.append(h)
        h = torch.cat(pooled, dim=1)        # (B, C * n_kernels)
        h = self.relu(self.fc1(h))          # (B, dense_dim)
        h = self.dropout(h)
        logit = self.fc2(h).squeeze(1)      # (B,)
        return logit


def pos_weight_from_labels(y) -> float:
    """neg / pos ratio, for ``BCEWithLogitsLoss(pos_weight=...)`` (1.0 if no positives)."""
    import numpy as np

    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0
