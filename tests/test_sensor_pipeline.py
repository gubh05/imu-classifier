"""Tests for SensorPipeline — end-to-end fit/predict/evaluate with dummy data."""
import numpy as np
import pytest

from src.features.feature_extractor import FeatureExtractor
from src.models.rf_baseline import RFClassifier
from src.pipeline.preprocessor import SignalPreprocessor
from src.pipeline.sensor_pipeline import SensorPipeline


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 200)
    return X, y


@pytest.fixture
def fitted_pipeline(dummy_data):
    X, y = dummy_data
    pipe = SensorPipeline(SignalPreprocessor(), FeatureExtractor(), RFClassifier())
    pipe.fit(X, y)
    return pipe, X, y


def test_fit_returns_self(dummy_data):
    X, y = dummy_data
    pipe = SensorPipeline(SignalPreprocessor(), FeatureExtractor(), RFClassifier())
    result = pipe.fit(X, y)
    assert result is pipe


def test_predict_shape(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    preds = pipe.predict(X[:30])
    assert preds.shape == (30,)


def test_predict_labels_in_range(fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    preds = pipe.predict(X)
    assert preds.min() >= 0 and preds.max() <= 5


def test_evaluate_returns_required_keys(fitted_pipeline):
    pipe, X, y = fitted_pipeline
    metrics = pipe.evaluate(X[:50], y[:50])
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics


def test_evaluate_accuracy_in_range(fitted_pipeline):
    pipe, X, y = fitted_pipeline
    metrics = pipe.evaluate(X[:50], y[:50])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_confusion_matrix_shape(fitted_pipeline):
    pipe, X, y = fitted_pipeline
    metrics = pipe.evaluate(X[:50], y[:50])
    cm = metrics["confusion_matrix"]
    assert cm.shape == (6, 6)


def test_save_and_load_round_trip(fitted_pipeline, tmp_path):
    pipe, X, _ = fitted_pipeline
    save_path = tmp_path / "pipeline.pkl"
    pipe.save(str(save_path))
    assert save_path.exists()

    loaded = SensorPipeline.load(str(save_path))
    preds_orig = pipe.predict(X[:10])
    preds_loaded = loaded.predict(X[:10])
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_preprocessor_fitted_after_fit(fitted_pipeline):
    pipe, _, _ = fitted_pipeline
    assert pipe.preprocessor._mean is not None
    assert pipe.preprocessor._std is not None
