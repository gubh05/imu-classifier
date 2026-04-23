"""Tests for RealTimeSimulator — window count, output format, edge cases."""
import numpy as np
import pytest

from src.features.feature_extractor import FeatureExtractor
from src.models.rf_baseline import RFClassifier
from src.pipeline.preprocessor import SignalPreprocessor
from src.pipeline.sensor_pipeline import SensorPipeline
from src.simulator.realtime_simulator import RealTimeSimulator


@pytest.fixture(scope="module")
def fitted_pipe():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 300)
    pipe = SensorPipeline(SignalPreprocessor(), FeatureExtractor(), RFClassifier())
    pipe.fit(X, y)
    return pipe


def _expected_windows(signal_len: int, window: int, step: int) -> int:
    return (signal_len - window) // step + 1


@pytest.mark.parametrize("signal_len,window,step", [
    (640, 128, 32),
    (512, 128, 64),
    (256, 128, 128),
    (128, 128, 32),   # exactly one window
])
def test_window_count(fitted_pipe, signal_len, window, step):
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((signal_len, 9))
    sim = RealTimeSimulator(fitted_pipe, window_size=window, step_size=step)
    results = list(sim.stream(raw))
    expected = _expected_windows(signal_len, window, step)
    assert len(results) == expected, f"Expected {expected} windows, got {len(results)}"


def test_output_tuple_structure(fitted_pipe):
    raw = np.random.randn(640, 9)
    sim = RealTimeSimulator(fitted_pipe)
    results = list(sim.stream(raw))
    for step_idx, label, conf in results:
        assert isinstance(step_idx, int)
        assert isinstance(label, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 100.0


def test_step_indices_sequential(fitted_pipe):
    raw = np.random.randn(640, 9)
    sim = RealTimeSimulator(fitted_pipe)
    indices = [r[0] for r in sim.stream(raw)]
    assert indices == list(range(len(indices)))


def test_labels_are_valid_activity_names(fitted_pipe):
    from src.utils.config import Config
    valid = set(Config().activity_labels.values())
    raw = np.random.randn(640, 9)
    sim = RealTimeSimulator(fitted_pipe)
    for _, label, _ in sim.stream(raw):
        assert label in valid, f"Unknown label: {label}"


def test_short_signal_raises(fitted_pipe):
    raw = np.random.randn(64, 9)   # shorter than window_size=128
    sim = RealTimeSimulator(fitted_pipe, window_size=128, step_size=32)
    with pytest.raises(ValueError):
        list(sim.stream(raw))


def test_wrong_channel_count_raises(fitted_pipe):
    raw = np.random.randn(640, 6)   # wrong channel count
    sim = RealTimeSimulator(fitted_pipe)
    with pytest.raises(ValueError):
        list(sim.stream(raw))
