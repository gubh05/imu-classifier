"""Tests for FeatureExtractor — output shape, feature name alignment, no NaNs."""
import numpy as np
import pytest

from src.features.feature_extractor import FeatureExtractor


@pytest.fixture
def fe():
    return FeatureExtractor()


@pytest.fixture
def dummy_X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 128, 9)).astype(np.float32)


def test_output_sample_count(fe, dummy_X):
    out = fe.extract(dummy_X)
    assert out.shape[0] == 50


def test_output_feature_count_matches_names(fe, dummy_X):
    out = fe.extract(dummy_X)
    names = fe.feature_names()
    assert out.shape[1] == len(names), (
        f"Feature matrix has {out.shape[1]} cols but feature_names() returns {len(names)}"
    )


def test_no_nans(fe, dummy_X):
    out = fe.extract(dummy_X)
    assert not np.isnan(out).any(), "NaN values found in feature matrix"


def test_no_infs(fe, dummy_X):
    out = fe.extract(dummy_X)
    assert not np.isinf(out).any(), "Inf values found in feature matrix"


def test_feature_names_unique(fe):
    names = fe.feature_names()
    assert len(names) == len(set(names)), "Feature names contain duplicates"


def test_feature_names_cover_all_channels(fe):
    names = fe.feature_names()
    for ch in ["body_acc_x", "body_gyro_z", "total_acc_y"]:
        assert any(n.startswith(ch) for n in names), f"Channel {ch} missing from feature names"


def test_single_sample(fe):
    X_single = np.random.randn(1, 128, 9).astype(np.float32)
    out = fe.extract(X_single)
    assert out.shape[0] == 1


def test_output_dtype_is_float(fe, dummy_X):
    out = fe.extract(dummy_X)
    assert np.issubdtype(out.dtype, np.floating), f"Expected float dtype, got {out.dtype}"
