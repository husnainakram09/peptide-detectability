"""Tests for the Gradio app's callbacks (``app.py``).

These run against the committed champion model; no Gradio server is started.
Validation rules themselves are covered in ``test_validation.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from src.validation import MAX_ROWS  # noqa: E402

VALID = "TITLEVEPSDTIENVK"


@pytest.fixture(scope="module")
def version():
    v = app._default_version()
    if v is None:
        pytest.skip("no trained model in models/")
    return v


def test_single_out_of_domain_is_not_scored(version):
    out = app.predict_single("XXXXXXXX", version)
    assert "Not scored" in out and "P(detectable)" not in out


def test_single_valid_is_scored(version):
    assert "P(detectable)" in app.predict_single(VALID.lower(), version)


def test_batch_marks_invalid_rows_and_scores_the_rest(version):
    df, csv_path, status = app.predict_batch(
        f"{VALID}\nX\n\nHQGVMVGMGQK\nSHORT", version
    )
    assert list(df["sequence"]) == [VALID, "X", "HQGVMVGMGQK", "SHORT"]
    scored = df["probability"].notna()
    assert list(scored) == [True, False, True, False]
    assert df.loc[~scored, "verdict"].str.startswith("not scored").all()
    assert df.loc[scored, "probability"].between(0, 1).all()
    assert "Scored 2 of 4" in status and Path(csv_path).exists()


def test_batch_over_limit_is_refused(version):
    df, csv_path, status = app.predict_batch("\n".join([VALID] * (MAX_ROWS + 1)), version)
    assert df.empty and csv_path is None and "limit" in status


def test_csv_uses_sequence_column_and_skips_blanks(tmp_path, version):
    path = tmp_path / "in.csv"
    pd.DataFrame({"id": [1, 2, 3], "Sequence": [VALID, None, "ESTLHLVLR"]}) \
        .to_csv(path, index=False)
    df, _, _ = app.predict_csv(str(path), version)
    assert list(df["sequence"]) == [VALID, "ESTLHLVLR"]
    assert df["probability"].notna().all()


def test_csv_over_limit_is_refused(tmp_path, version):
    path = tmp_path / "big.csv"
    pd.DataFrame({"sequence": [VALID] * (MAX_ROWS + 10)}).to_csv(path, index=False)
    df, _, status = app.predict_csv(str(path), version)
    assert df.empty and "limit" in status
