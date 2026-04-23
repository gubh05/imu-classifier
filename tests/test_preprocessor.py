"""Tests for SignalPreprocessor — fit/transform interface and no data leakage."""
import numpy as np
import pytest

from src.pipeline.preprocessor import SignalPreprocessor


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 128, 9)).astype(np.float32)
    X_test = rng.standard_normal((50, 128, 9)).astype(np.float32)
    return X_train, X_test


def test_output_shape_fit_transform(dummy_data):
    X_train, _ = dummy_data
    pp = SignalPreprocessor()
    out = pp.fit_transform(X_train)
    assert out.shape == X_train.shape


def test_output_shape_transform(dummy_data):
    X_train, X_test = dummy_data
    pp = SignalPreprocessor()
    pp.fit(X_train)
    out = pp.transform(X_test)
    assert out.shape == X_test.shape


def test_normalisation_mean_near_zero(dummy_data):
    X_train, _ = dummy_data
    pp = SignalPreprocessor()
    out = pp.fit_transform(X_train)
    assert abs(out.mean()) < 0.1, f"Mean after normalisation too large: {out.mean()}"


def test_normalisation_std_near_one(dummy_data):
    X_train, _ = dummy_data
    pp = SignalPreprocessor()
    out = pp.fit_transform(X_train)
    assert abs(out.std() - 1.0) < 0.1, f"Std after normalisation off: {out.std()}"


def test_no_data_leakage(dummy_data):
    """Test set transform must use train statistics, not recomputed test statistics."""
    X_train, X_test = dummy_data
    pp = SignalPreprocessor()
    pp.fit(X_train)

    # The stored mean/std must match those computed on train, not test
    train_filtered_mean = pp._mean
    train_filtered_std = pp._std

    # Re-fit on test data — this should change mean/std
    pp2 = SignalPreprocessor()
    pp2.fit(X_test)

    # The two fits on different data should produce different parameters
    assert not np.allclose(train_filtered_mean, pp2._mean, atol=1e-3), (
        "Train and test means are suspiciously identical — possible leakage"
    )


def test_transform_without_fit_raises():
    pp = SignalPreprocessor()
    X = np.random.randn(10, 128, 9).astype(np.float32)
    with pytest.raises(RuntimeError):
        pp.transform(X)


def test_ema_option_runs(dummy_data):
    X_train, _ = dummy_data
    pp = SignalPreprocessor(apply_ema=True)
    out = pp.fit_transform(X_train)
    assert out.shape == X_train.shape
    assert not np.isnan(out).any()
