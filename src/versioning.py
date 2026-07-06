# src/versioning.py
"""Append-only, never-overwrite model registry."""
from __future__ import annotations
import json, re, subprocess, hashlib, datetime
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REGISTRY = MODELS_DIR / "registry.json"
LATEST = MODELS_DIR / "LATEST.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def _next_index() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    idxs = []
    for p in MODELS_DIR.iterdir():
        if p.is_dir():
            m = re.match(r"v(\d+)_", p.name)
            if m:
                idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 1


def new_version_dir(model_type: str) -> Path:
    """Create and return a brand-new version directory. NEVER overwrites."""
    idx = _next_index()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"v{idx:03d}_{model_type}_{ts}"
    # Hard guard: if a dir with this name somehow exists, bump the index.
    while path.exists():
        idx += 1
        path = MODELS_DIR / f"v{idx:03d}_{model_type}_{ts}"
    path.mkdir(parents=True, exist_ok=False)  # fails loudly on any race
    return path


def register(version_dir: Path, model_type: str, metrics: dict[str, Any],
             config: dict[str, Any], data_hash: str) -> dict:
    entry = {
        "version": version_dir.name,
        "model_type": model_type,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data_hash": data_hash,
        "metrics": metrics,
        "path": str(version_dir.relative_to(MODELS_DIR.parent)),
    }
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (version_dir / "config.json").write_text(json.dumps(config, indent=2))
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    registry.append(entry)              # append-only
    REGISTRY.write_text(json.dumps(registry, indent=2))
    return entry


def update_champion(metric: str = "pr_auc", higher_is_better: bool = True) -> dict:
    """Recompute the champion pointer from the registry. Deletes nothing."""
    registry = json.loads(REGISTRY.read_text())
    best = None
    for e in registry:
        val = e["metrics"].get(metric)
        if val is None:
            continue
        if best is None or ((val > best["metrics"][metric]) == higher_is_better):
            best = e
    latest = {
        "champion": best["version"],
        "metric": metric,
        "value": best["metrics"][metric],
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    LATEST.write_text(json.dumps(latest, indent=2))
    return latest


def list_versions() -> list[dict]:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def champion_dir() -> Path:
    return MODELS_DIR / json.loads(LATEST.read_text())["champion"]


def data_hash_of(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]
