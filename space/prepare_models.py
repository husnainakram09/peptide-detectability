"""Copy trained model artifacts from the training repo (../models) into ./models.

Run this from the ``space/`` folder AFTER training locally, then push the Space::

    cd space
    python prepare_models.py                       # registry + LATEST + every version
    python prepare_models.py --champion-only        # just the champion
    python prepare_models.py --versions v001_baseline_... v003_esm2_...

The Gradio app loads models from ``space/models/`` (paths resolve relative to
``space/`` as the repo root), so at minimum copy the champion.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent          # space/
REPO_MODELS = HERE.parent / "models"            # ../models  (training repo)
SPACE_MODELS = HERE / "models"                  # ./models   (pushed to the Space)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy trained models into space/models/.")
    parser.add_argument("--champion-only", action="store_true",
                        help="Copy only the champion version directory.")
    parser.add_argument("--versions", nargs="*", default=None,
                        help="Explicit list of version directory names to copy.")
    args = parser.parse_args()

    registry = REPO_MODELS / "registry.json"
    latest = REPO_MODELS / "LATEST.json"
    if not registry.exists():
        raise SystemExit(
            f"{registry} not found — train a model first "
            f"(e.g. `python -m src.train --model baseline` from the repo root)."
        )

    SPACE_MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry, SPACE_MODELS / "registry.json")
    if latest.exists():
        shutil.copy2(latest, SPACE_MODELS / "LATEST.json")

    reg = json.loads(registry.read_text())
    all_versions = [e["version"] for e in reg]
    champion = json.loads(latest.read_text())["champion"] if latest.exists() else None

    if args.champion_only:
        chosen = [champion] if champion else []
    elif args.versions:
        chosen = args.versions
    else:
        chosen = all_versions

    copied = []
    for version in chosen:
        src = REPO_MODELS / version
        if src.is_dir():
            dst = SPACE_MODELS / version
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(version)
        else:
            print(f"  [skip] {version}: not found in {REPO_MODELS}")

    print(f"Copied registry (+ LATEST) and {len(copied)} version(s) into "
          f"{SPACE_MODELS}: {copied}")
    if champion:
        print(f"Champion: {champion}")
    if not copied:
        print("WARNING: no version directories were copied — the demo will have no "
              "model to select. Train first, then rerun.")


if __name__ == "__main__":
    main()
