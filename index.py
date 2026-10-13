"""JSON prediction API + static web page (Vercel entrypoint).

Vercel runs this FastAPI ``app`` as a serverless function; the page in ``web/``
is served from Vercel's CDN via ``app.frontend``. Every request is stateless:
the champion model is loaded once per warm instance and reused.

Endpoints:
    GET  /api/health   -> {"status": "ok", "model": <champion>}
    GET  /api/model    -> champion metadata + input rules
    POST /api/predict  -> {"sequences": [...]}  ->  per-sequence results

Run locally:  uvicorn index:app --reload   (then open http://127.0.0.1:8000)
The Gradio demo for Hugging Face Spaces lives in app.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src import predict as predictor
from src import validation

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="Peptide Detectability API",
    description=(
        "Predicts whether a tryptic peptide is likely to be detected in a "
        "bottom-up mass-spectrometry experiment. Research use only."
    ),
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)


class PredictRequest(BaseModel):
    sequences: list[str] = Field(
        ...,
        description="Peptide sequences (one-letter codes). Blank entries are ignored.",
        examples=[["TITLEVEPSDTIENVK", "HQGVMVGMGQK"]],
    )


class PredictionRow(BaseModel):
    sequence: str
    length: int
    probability: "float | None" = Field(
        description="P(detectable), or null when the sequence was not scored."
    )
    verdict: str


class PredictResponse(BaseModel):
    model: str
    threshold: float
    n_scored: int
    message: str
    results: list[PredictionRow]


@lru_cache(maxsize=1)
def _champion() -> "tuple[str, callable]":
    """Load the champion model once per warm instance."""
    version = predictor.champion_version()
    if not version:
        versions = predictor.list_versions()
        version = versions[-1]["version"] if versions else None
    if not version:
        raise RuntimeError("No trained model found in models/.")
    for artifact in (config.MODELS_DIR / version).iterdir():
        if artifact.suffix not in {".pkl", ".npz", ".pt"}:
            continue
        with open(artifact, "rb") as fh:
            if fh.read(24).startswith(b"version https://git-lfs"):
                raise RuntimeError(
                    f"{artifact.name} is a Git LFS pointer, not the model. "
                    "Enable Git LFS in the Vercel project settings (Git) and redeploy."
                )
    return version, predictor.load_model(version)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": predictor.champion_version()}


@app.get("/api/model")
def model_info() -> dict:
    version = predictor.champion_version()
    entry = next(
        (e for e in predictor.list_versions() if e["version"] == version), {}
    )
    metrics = entry.get("metrics", {})
    return {
        "version": version,
        "model_type": entry.get("model_type"),
        "test_metrics": {
            key: metrics[key]
            for key in ("pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1")
            if key in metrics
        },
        "threshold": validation.THRESHOLD,
        "max_sequences": validation.MAX_ROWS,
        "min_length": config.MIN_LEN,
        "max_length": config.MAX_LEN,
        "alphabet": config.AMINO_ACIDS,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    try:
        version, predict_fn = _champion()
    except Exception as exc:
        # Surface the cause (e.g. Git LFS pointers on Vercel) instead of a bare 500.
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    try:
        rows = validation.score_sequences(request.sequences, predict_fn)
    except validation.TooManySequences as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if not rows:
        raise HTTPException(status_code=422, detail="No sequences provided.")
    return {
        "model": version,
        "threshold": validation.THRESHOLD,
        "n_scored": sum(row["probability"] is not None for row in rows),
        "message": validation.summary(rows),
        "results": rows,
    }


# Registered last: API routes always take priority over frontend files.
app.frontend("/", directory=WEB_DIR)
