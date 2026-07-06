"""Central configuration: paths, constants, and the amino-acid vocabulary.

This module holds only *static* configuration for the peptide-detectability
project. It contains no model, data, or training logic (those arrive in later
phases) and has no import-time side effects (it never creates directories).

Target: Python 3.11, CPU-only, all dependencies free / open-source.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
# PROJECT_ROOT is the repository root — the directory that contains ``src/``.
# ``config.py`` lives at ``<root>/src/config.py``; parents[1] is ``<root>``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Cache for ESM-2 embeddings (created on demand in Phase 5; git-ignored).
EMBEDDINGS_CACHE_DIR: Path = PROJECT_ROOT / "embeddings_cache"

# Convenience handles for the processed split files (created in Phase 1).
TRAIN_CSV: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV: Path = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV: Path = PROCESSED_DATA_DIR / "test.csv"
SPLIT_STATS_JSON: Path = PROCESSED_DATA_DIR / "split_stats.json"

# --------------------------------------------------------------------------- #
# Amino-acid vocabulary
# --------------------------------------------------------------------------- #
# The 20 standard amino acids in a fixed canonical order.
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"

# Index 0 is reserved for padding; real amino acids are indexed starting at 1.
PAD_IDX: int = 0
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS, start=1)}
IDX_TO_AA: dict[int, str] = {i: aa for aa, i in AA_TO_IDX.items()}

# Number of real tokens (== 20). Embedding layers should be sized
# ``VOCAB_SIZE + 1`` so that the padding index (0) has its own row.
VOCAB_SIZE: int = len(AMINO_ACIDS)

# --------------------------------------------------------------------------- #
# Sequence-length bounds (inclusive)
# --------------------------------------------------------------------------- #
MAX_LEN: int = 50
MIN_LEN: int = 7

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# Primary metric used to select the champion model (higher is better). Chosen
# because detectability datasets are class-imbalanced, where PR-AUC is more
# informative than accuracy.
PRIMARY_METRIC: str = "pr_auc"
