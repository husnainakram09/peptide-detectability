"""Tests for the shared input validation / scoring helpers (``src/validation.py``)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import validation  # noqa: E402

VALID = "TITLEVEPSDTIENVK"


def _fake_predict(sequences):
    """Deterministic stand-in for a model: P = length / 100."""
    return [len(s) / 100 for s in sequences]


def test_validate_accepts_in_domain_peptide():
    assert validation.validate_sequence(VALID) is None


@pytest.mark.parametrize("seq, reason", [
    ("PEPTIDEXK", "non-standard"),
    ("PEPBJOUZK", "non-standard"),
    ("PEPM(OX)K", "non-standard"),
    ("PEPTIK", "length 6"),
    ("A" * 51, "length 51"),
])
def test_validate_rejects_out_of_domain(seq, reason):
    assert reason in validation.validate_sequence(seq)


def test_clean_sequence_strips_whitespace_but_keeps_other_characters():
    assert validation.clean_sequence("  titl evep\tsdtienvk ") == VALID
    assert validation.clean_sequence("pep-tide") == "PEP-TIDE"


def test_score_sequences_marks_invalid_rows_and_scores_the_rest():
    rows = validation.score_sequences([VALID, "X", "", "  ", "HQGVMVGMGQK"], _fake_predict)
    assert [r["sequence"] for r in rows] == [VALID, "X", "HQGVMVGMGQK"]
    assert [r["probability"] for r in rows] == [0.16, None, 0.11]
    assert rows[1]["verdict"].startswith("not scored")
    assert rows[0]["verdict"] == "likely OVERLOOKED"


def test_score_sequences_does_not_call_model_when_nothing_is_valid():
    def explode(_):
        raise AssertionError("model should not be called")

    rows = validation.score_sequences(["X", "SHORT"], explode)
    assert all(r["probability"] is None for r in rows)


def test_score_sequences_enforces_row_limit():
    with pytest.raises(validation.TooManySequences):
        validation.score_sequences([VALID] * (validation.MAX_ROWS + 1), _fake_predict)


def test_summary_mentions_unscored_rows():
    rows = validation.score_sequences([VALID, "X"], _fake_predict)
    message = validation.summary(rows)
    assert "Scored 1 of 2" in message and "not scored" in message
