"""NumPy inference for ``PeptideCNN`` — no PyTorch needed at prediction time.

Mirrors ``PeptideCNN.forward`` (``src/models/cnn.py``) operation for operation::

    embedding lookup -> per-kernel Conv1d ("same" padding) -> ReLU
      -> global max-pool over positions (padding included, as in PyTorch)
      -> concat -> Linear -> ReLU -> Linear  [logit]

Dropout is a no-op at inference. Weights are exported from the PyTorch state
dict to ``model.npz`` with ``export_weights``; ``tests/test_cnn_numpy.py``
checks that both implementations return the same probabilities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def export_weights(state_dict, path: Path) -> None:
    """Save a ``PeptideCNN`` state dict as a NumPy ``.npz`` archive."""
    np.savez(path, **{k: v.detach().cpu().numpy() for k, v in state_dict.items()})


class NumpyPeptideCNN:
    """Forward pass of ``PeptideCNN`` using exported weights."""

    def __init__(self, weights: dict, kernel_sizes) -> None:
        self.embedding = weights["embedding.weight"]
        self.convs = [
            (weights[f"convs.{i}.weight"], weights[f"convs.{i}.bias"], int(k))
            for i, k in enumerate(kernel_sizes)
        ]
        self.fc1_w, self.fc1_b = weights["fc1.weight"], weights["fc1.bias"]
        self.fc2_w, self.fc2_b = weights["fc2.weight"], weights["fc2.bias"]

    @classmethod
    def load(cls, path: Path, kernel_sizes) -> "NumpyPeptideCNN":
        with np.load(path) as archive:
            return cls({name: archive[name] for name in archive.files}, kernel_sizes)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """``x``: ``(B, L)`` integer token ids -> ``(B,)`` logits."""
        emb = self.embedding[x]                                     # (B, L, E)
        pooled = []
        for weight, bias, k in self.convs:                          # weight: (C, E, k)
            pad = k // 2
            padded = np.pad(emb, ((0, 0), (pad, pad), (0, 0)))
            windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)
            windows = windows.reshape(*windows.shape[:2], -1)       # (B, L, E*k)
            out = windows @ weight.reshape(weight.shape[0], -1).T + bias
            pooled.append(np.maximum(out, 0).max(axis=1))           # (B, C)
        h = np.concatenate(pooled, axis=1)
        h = np.maximum(h @ self.fc1_w.T + self.fc1_b, 0)
        return (h @ self.fc2_w.T + self.fc2_b)[:, 0]

    def predict_proba(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """P(detectable) for ``(B, L)`` token ids, processed in batches."""
        if len(x) == 0:
            return np.zeros(0, dtype=float)
        logits = np.concatenate([
            self.logits(x[i:i + batch_size]) for i in range(0, len(x), batch_size)
        ])
        return (0.5 * (1.0 + np.tanh(0.5 * logits))).astype(float)  # stable sigmoid
