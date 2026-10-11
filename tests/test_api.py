"""Tests for the Vercel FastAPI entrypoint (``index.py``) using TestClient."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import index  # noqa: E402
from src.validation import MAX_ROWS  # noqa: E402

VALID = "TITLEVEPSDTIENVK"


@pytest.fixture(scope="module")
def client():
    if index.predictor.champion_version() is None:
        pytest.skip("no trained model in models/")
    return TestClient(index.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_exposes_input_rules(client):
    body = client.get("/api/model").json()
    assert body["version"] and body["max_sequences"] == MAX_ROWS
    assert (body["min_length"], body["max_length"]) == (7, 50)
    assert 0 < body["test_metrics"]["pr_auc"] <= 1


def test_predict_scores_valid_and_flags_invalid(client):
    r = client.post("/api/predict", json={"sequences": [VALID, "pep-x", " "]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_scored"] == 1
    assert [row["sequence"] for row in body["results"]] == [VALID, "PEP-X"]
    assert 0 <= body["results"][0]["probability"] <= 1
    assert body["results"][1]["probability"] is None


def test_predict_matches_gradio_app(client):
    """Both front ends must return the same probability for the same input."""
    import app

    api = client.post("/api/predict", json={"sequences": [VALID]}).json()
    df, _, _ = app.predict_batch(VALID, app._default_version())
    assert api["results"][0]["probability"] == df["probability"].iloc[0]


def test_predict_rejects_empty_and_oversized_requests(client):
    assert client.post("/api/predict", json={"sequences": ["", "  "]}).status_code == 422
    assert client.post("/api/predict", json={"sequences": []}).status_code == 422
    assert client.post("/api/predict", json={}).status_code == 422
    r = client.post("/api/predict", json={"sequences": [VALID] * (MAX_ROWS + 1)})
    assert r.status_code == 413 and "limit" in r.json()["detail"]


def test_frontend_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200 and "Peptide Detectability Predictor" in r.text
