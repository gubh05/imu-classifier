"""Tests for RF, CNN, and LSTM models — uses small dummy data, no real dataset."""
import numpy as np
import pytest
import torch

from src.models.rf_baseline import RFClassifier
from src.models.cnn_classifier import CNNClassifier, DEVICE
from src.models.lstm_classifier import LSTMClassifier


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

@pytest.fixture
def rf_fitted():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 117)).astype(np.float32)
    y = rng.integers(0, 6, 200)
    clf = RFClassifier()
    clf.fit(X, y)
    return clf, X, y


def test_rf_predict_shape(rf_fitted):
    clf, X, _ = rf_fitted
    preds = clf.predict(X[:20])
    assert preds.shape == (20,)


def test_rf_predict_values_in_range(rf_fitted):
    clf, X, _ = rf_fitted
    preds = clf.predict(X)
    assert preds.min() >= 0 and preds.max() <= 5


def test_rf_predict_proba_shape(rf_fitted):
    clf, X, _ = rf_fitted
    proba = clf.predict_proba(X[:10])
    assert proba.shape == (10, 6)


def test_rf_predict_proba_sums_to_one(rf_fitted):
    clf, X, _ = rf_fitted
    proba = clf.predict_proba(X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_rf_feature_importances(rf_fitted):
    clf, X, _ = rf_fitted
    imp = clf.feature_importances_
    assert imp.shape == (117,)
    assert abs(imp.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

def test_cnn_forward_shape():
    model = CNNClassifier()
    x = torch.randn(8, 9, 128).to(DEVICE)
    out = model(x)
    assert out.shape == (8, 6), f"Expected (8,6), got {out.shape}"


def test_cnn_forward_no_nan():
    model = CNNClassifier()
    x = torch.randn(4, 9, 128).to(DEVICE)
    out = model(x)
    assert not torch.isnan(out).any()


def test_cnn_fit_predict():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 60)
    clf = CNNClassifier(epochs=2)
    clf.fit(X, y)
    preds = clf.predict(X[:10])
    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 5


def test_cnn_predict_proba_shape():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 30)
    clf = CNNClassifier(epochs=1)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 6)


def test_cnn_train_losses_recorded():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 40)
    clf = CNNClassifier(epochs=3)
    clf.fit(X, y)
    assert len(clf.train_losses) == 3
    assert len(clf.train_accs) == 3


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def test_lstm_forward_shape():
    model = LSTMClassifier()
    x = torch.randn(8, 128, 9).to(DEVICE)
    out = model(x)
    assert out.shape == (8, 6), f"Expected (8,6), got {out.shape}"


def test_lstm_forward_no_nan():
    model = LSTMClassifier()
    x = torch.randn(4, 128, 9).to(DEVICE)
    out = model(x)
    assert not torch.isnan(out).any()


def test_lstm_fit_predict():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 60)
    clf = LSTMClassifier(epochs=2)
    clf.fit(X, y)
    preds = clf.predict(X[:10])
    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 5


def test_lstm_predict_proba_shape():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((30, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 30)
    clf = LSTMClassifier(epochs=1)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 6)


def test_lstm_train_losses_recorded():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((40, 128, 9)).astype(np.float32)
    y = rng.integers(0, 6, 40)
    clf = LSTMClassifier(epochs=3)
    clf.fit(X, y)
    assert len(clf.train_losses) == 3
    assert len(clf.train_accs) == 3
