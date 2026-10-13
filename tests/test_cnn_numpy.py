"""The NumPy CNN forward pass must match the PyTorch ``PeptideCNN`` exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src import encoders  # noqa: E402
from src.models import cnn, cnn_numpy  # noqa: E402

SEQUENCES = ["TITLEVEPSDTIENVK", "HQGVMVGMGQK", "ESTLHLVLR", "K" * 50, "ACDEFGH"]


@pytest.mark.parametrize("kernel_sizes", [(3, 5), (3, 5, 7)])
def test_numpy_forward_matches_pytorch(tmp_path, kernel_sizes):
    torch.manual_seed(0)
    model = cnn.PeptideCNN(vocab_size=20, kernel_sizes=kernel_sizes).eval()
    cnn_numpy.export_weights(model.state_dict(), tmp_path / "model.npz")
    np_model = cnn_numpy.NumpyPeptideCNN.load(tmp_path / "model.npz", kernel_sizes)

    x = encoders.encode_sequences(SEQUENCES)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(x))).numpy()

    np.testing.assert_allclose(np_model.predict_proba(x, batch_size=2), expected, atol=1e-5)


def test_empty_input():
    torch.manual_seed(0)
    model = cnn.PeptideCNN(vocab_size=20)
    weights = {k: v.numpy() for k, v in model.state_dict().items()}
    np_model = cnn_numpy.NumpyPeptideCNN(weights, (3, 5))
    assert np_model.predict_proba(np.zeros((0, 50), dtype=np.int64)).shape == (0,)
