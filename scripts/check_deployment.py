"""Check a running deployment against the labelled held-out test set.

Sends data/processed/test.csv to <url>/api/predict and compares the verdicts
with the true labels. A correct deployment reproduces the champion's row in
reports/benchmark.md (accuracy 0.7854, precision 0.7578, recall 0.8453 for
v003_cnn).

Uses only the standard library, so it runs from any Python 3.9+.

    python scripts/check_deployment.py http://127.0.0.1:8000
    python scripts/check_deployment.py https://<your-project>.vercel.app
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TEST_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.csv"
BATCH = 2000  # stay well under the 5,000-sequence / 4.5 MB request limits


def post(url: str, sequences: list[str]) -> dict:
    request = urllib.request.Request(
        f"{url}/api/predict",
        data=json.dumps({"sequences": sequences}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="Base URL, e.g. https://my-app.vercel.app")
    parser.add_argument("--csv", type=Path, default=TEST_CSV,
                        help="CSV with 'sequence' and 'label' columns (default: test split)")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    if not args.csv.exists():
        sys.exit(f"{args.csv} not found. Run `python -m src.data_prep` first.")
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    try:
        health = json.load(urllib.request.urlopen(f"{url}/api/health", timeout=60))
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach {url}: {exc}")
    print(f"Server model : {health.get('model')}")
    print(f"Test peptides: {len(rows):,} from {args.csv.name}")

    results = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        results += post(url, [r["sequence"] for r in chunk])["results"]

    tp = fp = tn = fn = skipped = 0
    for row, result in zip(rows, results):
        if result["probability"] is None:
            skipped += 1
            continue
        predicted = result["probability"] >= 0.5
        actual = row["label"] == "1"
        tp += predicted and actual
        fp += predicted and not actual
        tn += not predicted and not actual
        fn += not predicted and actual

    scored = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"Scored       : {scored:,}  (not scored: {skipped})")
    print(f"Accuracy     : {(tp + tn) / scored:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"Confusion    : TN={tn}  FP={fp}  FN={fn}  TP={tp}")


if __name__ == "__main__":
    main()
