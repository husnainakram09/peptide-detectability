"""Physicochemical + composition features for the baseline model.

Feature vector per peptide (length 1 + 20 + 6 = 27), in this fixed order:

  index 0        : sequence length
  index 1..20    : amino-acid composition — 20 fractions, one per standard AA
                   (in ``config.AMINO_ACIDS`` order)
  index 21..26   : Biopython ProtParam descriptors —
                   molecular weight, GRAVY (hydrophobicity), aromaticity,
                   instability index, isoelectric point, net charge at pH 7.0.

The ordered feature names live in ``FEATURE_NAMES`` and the count in
``N_FEATURES`` so that downstream code (training config, prediction) can rely on
a stable layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from src import config

_AA = config.AMINO_ACIDS
_AA_INDEX = {aa: i for i, aa in enumerate(_AA)}
_CHARGE_PH = 7.0

# Names of the physicochemical descriptors, in output order.
_PHYSICO_NAMES = [
    "molecular_weight",
    "gravy",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    f"net_charge_pH{_CHARGE_PH:g}",
]

FEATURE_NAMES: list[str] = (
    ["length"] + [f"frac_{aa}" for aa in _AA] + _PHYSICO_NAMES
)
N_FEATURES = len(FEATURE_NAMES)  # 27
_PHYSICO_START = 1 + len(_AA)


def _sanitize(sequence: str) -> str:
    """Uppercase and keep only the 20 standard amino acids."""
    return "".join(c for c in str(sequence).strip().upper() if c in _AA_INDEX)


def _net_charge(pa: ProteinAnalysis) -> float:
    """Net charge at pH 7.0 (Biopython >= 1.72 exposes ``charge_at_pH``)."""
    try:
        return float(pa.charge_at_pH(_CHARGE_PH))
    except Exception:
        return 0.0


def featurize(sequence: str) -> np.ndarray:
    """Return the 27-dim feature vector for a single peptide sequence.

    Invalid / non-standard characters are dropped; an empty or fully-invalid
    sequence yields an all-zero vector (so the app never crashes on bad input).
    """
    seq = _sanitize(sequence)
    vec = np.zeros(N_FEATURES, dtype=np.float64)
    if not seq:
        return vec

    length = len(seq)
    vec[0] = float(length)

    # amino-acid composition as fractions
    for c in seq:
        vec[1 + _AA_INDEX[c]] += 1.0
    vec[1:_PHYSICO_START] /= length

    # physicochemical descriptors (Biopython ProtParam)
    pa = ProteinAnalysis(seq)
    try:
        mw = float(pa.molecular_weight())
    except Exception:
        mw = 0.0
    vec[_PHYSICO_START + 0] = mw
    vec[_PHYSICO_START + 1] = float(pa.gravy())
    vec[_PHYSICO_START + 2] = float(pa.aromaticity())
    vec[_PHYSICO_START + 3] = float(pa.instability_index())
    vec[_PHYSICO_START + 4] = float(pa.isoelectric_point())
    vec[_PHYSICO_START + 5] = _net_charge(pa)
    return vec


def featurize_many(sequences, progress: bool = False) -> np.ndarray:
    """Vectorize an iterable/Series of sequences into a ``(n, N_FEATURES)`` array."""
    seqs = list(sequences)
    indices = range(len(seqs))
    if progress:
        from tqdm import tqdm
        indices = tqdm(indices, desc="featurizing", unit="pep")
    out = np.zeros((len(seqs), N_FEATURES), dtype=np.float64)
    for i in indices:
        out[i] = featurize(seqs[i])
    return out


def featurize_dataframe(df: pd.DataFrame, seq_col: str = "sequence",
                        progress: bool = False) -> np.ndarray:
    """Vectorize a whole dataframe's sequence column into a feature array."""
    return featurize_many(df[seq_col].tolist(), progress=progress)
