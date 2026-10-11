"""Input validation and scoring shared by the two front ends.

Both the Gradio app (``app.py``, Hugging Face Space) and the JSON API
(``index.py``, Vercel) score peptides through ``score_sequences`` so that they
accept, reject and label sequences identically.

Only peptides inside the model's training domain — the 20 standard amino acids
and length ``config.MIN_LEN``..``config.MAX_LEN`` — are scored. Anything else is
returned with ``probability=None`` and a "not scored: <reason>" verdict instead
of a meaningless probability.
"""
from __future__ import annotations

from src import config

THRESHOLD = 0.5
MAX_ROWS = 5000  # per request — keeps a small CPU host responsive

_AA_SET = set(config.AMINO_ACIDS)


class TooManySequences(ValueError):
    """Raised when a request exceeds ``MAX_ROWS`` sequences."""


def clean_sequence(sequence) -> str:
    """Uppercase and remove all whitespace. Other characters are kept so that
    they can be reported by ``validate_sequence`` rather than silently dropped."""
    return "".join(str(sequence).split()).upper()


def validate_sequence(sequence: str) -> "str | None":
    """Return why ``sequence`` cannot be scored, or None if it is in-domain."""
    bad = sorted(set(sequence) - _AA_SET)
    if bad:
        return f"non-standard character(s) {''.join(bad)}"
    if not config.MIN_LEN <= len(sequence) <= config.MAX_LEN:
        return (
            f"length {len(sequence)} is outside "
            f"{config.MIN_LEN}–{config.MAX_LEN}"
        )
    return None


def verdict(probability: float) -> str:
    """Return the human-readable verdict for a probability."""
    return "likely detectable" if probability >= THRESHOLD else "likely OVERLOOKED"


def domain_note() -> str:
    """One-sentence description of the inputs the model accepts."""
    return (
        f"The model only covers peptides of {config.MIN_LEN}–{config.MAX_LEN} "
        "residues written with the 20 standard amino acids (no modifications)."
    )


def score_sequences(sequences, predict) -> "list[dict]":
    """Clean, validate and score ``sequences`` with ``predict``.

    ``predict`` is a callable ``list[str] -> array of P(detectable)`` as returned
    by ``src.predict.load_model``. Blank entries are dropped. Returns one dict
    per remaining sequence with keys ``sequence``, ``length``, ``probability``
    (None when not scored) and ``verdict``.

    Raises ``TooManySequences`` if more than ``MAX_ROWS`` sequences remain.
    """
    cleaned = [clean_sequence(sequence) for sequence in sequences]
    cleaned = [sequence for sequence in cleaned if sequence]

    if len(cleaned) > MAX_ROWS:
        raise TooManySequences(
            f"Too many sequences ({len(cleaned):,}). "
            f"The limit is {MAX_ROWS:,} per request."
        )

    problems = [validate_sequence(sequence) for sequence in cleaned]
    valid = [s for s, problem in zip(cleaned, problems) if problem is None]
    probabilities = iter(predict(valid) if valid else [])

    rows = []
    for sequence, problem in zip(cleaned, problems):
        if problem is None:
            probability = float(next(probabilities))
            rows.append({
                "sequence": sequence,
                "length": len(sequence),
                "probability": round(probability, 4),
                "verdict": verdict(probability),
            })
        else:
            rows.append({
                "sequence": sequence,
                "length": len(sequence),
                "probability": None,
                "verdict": f"not scored: {problem}",
            })
    return rows


def summary(rows: "list[dict]") -> str:
    """Markdown-free one-line status for a batch of scored rows."""
    n_scored = sum(row["probability"] is not None for row in rows)
    message = f"Scored {n_scored:,} of {len(rows):,} sequence(s)."
    if n_scored < len(rows):
        message += f" {len(rows) - n_scored:,} not scored. {domain_note()}"
    return message
