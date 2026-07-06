"""Tests for the never-overwrite versioning registry (``src/versioning.py``).

Each test runs against a fresh temporary ``models/`` directory: we monkeypatch
``versioning.MODELS_DIR`` / ``REGISTRY`` / ``LATEST`` so nothing touches the real
registry, and every check confirms the append-only / never-delete contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``from src import versioning`` importable when pytest is invoked from the
# repo root as either ``pytest tests/...`` or ``python -m pytest``. (The ``-m``
# form already puts the cwd on sys.path; this line covers the plain console form.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import versioning  # noqa: E402


@pytest.fixture()
def vers(tmp_path, monkeypatch):
    """Point the versioning module at an isolated, temporary ``models/`` dir."""
    models_dir = tmp_path / "models"
    monkeypatch.setattr(versioning, "MODELS_DIR", models_dir)
    monkeypatch.setattr(versioning, "REGISTRY", models_dir / "registry.json")
    monkeypatch.setattr(versioning, "LATEST", models_dir / "LATEST.json")
    return versioning


# (a) new_version_dir returns a fresh directory each call, never an existing one.
def test_new_version_dir_is_always_fresh(vers):
    d1 = vers.new_version_dir("baseline")
    assert d1.exists() and d1.is_dir()
    assert not any(d1.iterdir())  # brand-new and empty

    d2 = vers.new_version_dir("baseline")
    assert d2.exists() and d2.is_dir()
    assert d1 != d2  # a second call never hands back the same (existing) dir


# (a, extra) Hard guard: even if a higher-index dir already exists, it is never reused.
def test_new_version_dir_never_reuses_existing(vers):
    d1 = vers.new_version_dir("baseline")               # v001
    (vers.MODELS_DIR / "v009_manual_20200101_000000").mkdir()
    d_next = vers.new_version_dir("cnn")                 # must jump past v009
    assert d_next != d1
    assert d_next.name.startswith("v010_")


# (b) Two calls yield different dirs with incrementing v001, v002 indices.
def test_incrementing_indices(vers):
    d1 = vers.new_version_dir("baseline")
    d2 = vers.new_version_dir("cnn")
    assert d1.name.startswith("v001_")
    assert d2.name.startswith("v002_")
    assert d1 != d2


# (c) register appends to registry.json without dropping earlier entries.
def test_register_is_append_only(vers):
    d1 = vers.new_version_dir("baseline")
    vers.register(d1, "baseline", {"pr_auc": 0.80}, {"seed": 42}, "hash_a")
    d2 = vers.new_version_dir("cnn")
    vers.register(d2, "cnn", {"pr_auc": 0.90}, {"seed": 42}, "hash_b")

    registry = json.loads(vers.REGISTRY.read_text())
    assert len(registry) == 2
    versions = [entry["version"] for entry in registry]
    assert versions == [d1.name, d2.name]  # order preserved, nothing dropped

    first = registry[0]
    assert first["model_type"] == "baseline"
    assert first["metrics"]["pr_auc"] == 0.80
    assert first["data_hash"] == "hash_a"

    # per-version artifacts are written inside the version dir
    assert (d1 / "metrics.json").exists()
    assert (d1 / "config.json").exists()


# (d) update_champion writes LATEST.json but deletes no version directory.
def test_update_champion_writes_pointer_and_deletes_nothing(vers):
    d1 = vers.new_version_dir("baseline")
    vers.register(d1, "baseline", {"pr_auc": 0.70}, {}, "h1")
    d2 = vers.new_version_dir("cnn")
    vers.register(d2, "cnn", {"pr_auc": 0.95}, {}, "h2")

    latest = vers.update_champion(metric="pr_auc")
    assert vers.LATEST.exists()
    saved = json.loads(vers.LATEST.read_text())
    assert saved["champion"] == d2.name        # higher pr_auc wins
    assert saved["metric"] == "pr_auc"
    assert saved["value"] == 0.95
    assert latest == saved

    # nothing deleted; champion pointer resolves to a real, existing dir
    assert d1.exists() and d2.exists()
    assert vers.champion_dir() == d2


# (e) After registering two models, both version folders still exist on disk.
def test_both_versions_persist_on_disk(vers):
    d1 = vers.new_version_dir("baseline")
    vers.register(d1, "baseline", {"pr_auc": 0.60}, {}, "h1")
    d2 = vers.new_version_dir("cnn")
    vers.register(d2, "cnn", {"pr_auc": 0.75}, {}, "h2")
    vers.update_champion(metric="pr_auc")

    entries = vers.list_versions()
    assert len(entries) == 2
    for entry in entries:
        assert (vers.MODELS_DIR / entry["version"]).exists()
    assert d1.exists() and d2.exists()
