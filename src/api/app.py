"""FastAPI server for IMU activity classifier inference."""
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator

from src.models.cnn_classifier import CNNClassifier
from src.models.lstm_classifier import LSTMClassifier
from src.pipeline.sensor_pipeline import SensorPipeline
from src.utils.config import Config

logger = logging.getLogger(__name__)

app = FastAPI(title="IMU Activity Classifier API")
config = Config()

_PIPELINES: dict[str, SensorPipeline] = {}

OUTPUTS = Path("outputs")


def _build_deep_pipeline(model_name: str) -> SensorPipeline:
    """Reconstruct a full SensorPipeline for CNN or LSTM from saved artefacts.

    Borrows the fitted preprocessor and extractor from pipeline_rf.pkl, then
    loads the deep model weights and caches the assembled pipeline to disk.
    """
    rf_pkl = OUTPUTS / "pipeline_rf.pkl"
    if not rf_pkl.exists():
        raise FileNotFoundError(f"Required base pipeline not found: {rf_pkl}")

    rf_pipe: SensorPipeline = joblib.load(rf_pkl)
    preprocessor = rf_pipe.preprocessor
    extractor = rf_pipe.extractor

    pt_path = OUTPUTS / f"{model_name}_best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {pt_path}")

    import torch

    if model_name == "cnn":
        model = CNNClassifier()
        model.model.load_state_dict(
            torch.load(pt_path, map_location=model.device, weights_only=True)
        )
    else:  # lstm
        model = LSTMClassifier()
        model.model.load_state_dict(
            torch.load(pt_path, map_location=model.device, weights_only=True)
        )

    pipeline = SensorPipeline(preprocessor, extractor, model)
    cache_path = OUTPUTS / f"pipeline_{model_name}.pkl"
    joblib.dump(pipeline, cache_path)
    logger.info("Built and cached %s pipeline to %s", model_name, cache_path)
    return pipeline


def _load_pipeline(model_name: str) -> SensorPipeline:
    """Load a SensorPipeline from cache pkl, building it first if needed."""
    pkl_path = OUTPUTS / f"pipeline_{model_name}.pkl"
    if pkl_path.exists():
        return joblib.load(pkl_path)
    if model_name == "rf":
        alt = OUTPUTS / "pipeline_rf.pkl"
        if alt.exists():
            return joblib.load(alt)
        raise FileNotFoundError(f"RF pipeline not found at {alt}")
    return _build_deep_pipeline(model_name)


@app.on_event("startup")
def load_models() -> None:
    """Load all three pipelines at server startup."""
    for name in ("rf", "cnn", "lstm"):
        try:
            _PIPELINES[name] = _load_pipeline(name)
            logger.info("Loaded pipeline: %s", name)
        except Exception as exc:
            logger.warning("Could not load pipeline '%s': %s", name, exc)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    signal: list[list[float]]

    @field_validator("signal")
    @classmethod
    def validate_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 128:
            raise ValueError(f"signal must have 128 timesteps, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != 9:
                raise ValueError(
                    f"Each timestep must have 9 channels; row {i} has {len(row)}"
                )
        return v


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    model: str = Query(default="rf", pattern="^(rf|cnn|lstm)$"),
) -> PredictResponse:
    if model not in _PIPELINES:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is not loaded. Available: {list(_PIPELINES)}",
        )

    pipeline = _PIPELINES[model]
    X = np.array(body.signal, dtype=np.float32)[np.newaxis, ...]  # (1, 128, 9)

    try:
        t_start = time.perf_counter()
        proba = pipeline.model.predict_proba(pipeline._prepare_X(X))
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        class_idx = int(np.argmax(proba[0]))
        confidence = float(proba[0][class_idx])
        label = config.activity_labels.get(class_idx, str(class_idx))
    except Exception as exc:
        logger.exception("Inference error for model '%s': %s", model, exc)
        raise HTTPException(status_code=500, detail="Inference failed.")

    return PredictResponse(
        prediction=label,
        confidence=confidence,
        model=model,
        latency_ms=round(latency_ms, 3),
    )
