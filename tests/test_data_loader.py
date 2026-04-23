"""Tests for UCIHARDataLoader — uses cached data, no re-download."""
import numpy as np
import pytest

from src.ingestion.data_loader import UCIHARDataLoader
from src.utils.config import Config


@pytest.fixture(scope="module")
def loaded_data():
    loader = UCIHARDataLoader(Config())
    return loader.load()


def test_train_shape(loaded_data):
    X_train, _, _, _ = loaded_data
    assert X_train.shape == (7352, 128, 9), f"Unexpected train shape: {X_train.shape}"


def test_test_shape(loaded_data):
    _, X_test, _, _ = loaded_data
    assert X_test.shape == (2947, 128, 9), f"Unexpected test shape: {X_test.shape}"


def test_label_range_train(loaded_data):
    _, _, y_train, _ = loaded_data
    assert y_train.min() == 0, f"y_train min should be 0, got {y_train.min()}"
    assert y_train.max() == 5, f"y_train max should be 5, got {y_train.max()}"


def test_label_range_test(loaded_data):
    _, _, _, y_test = loaded_data
    assert y_test.min() == 0, f"y_test min should be 0, got {y_test.min()}"
    assert y_test.max() == 5, f"y_test max should be 5, got {y_test.max()}"


def test_no_nans(loaded_data):
    X_train, X_test, y_train, y_test = loaded_data
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(X_test).any(), "NaN in X_test"


def test_all_six_classes_present(loaded_data):
    _, _, y_train, y_test = loaded_data
    assert set(np.unique(y_train)) == {0, 1, 2, 3, 4, 5}
    assert set(np.unique(y_test)) == {0, 1, 2, 3, 4, 5}


def test_download_skipped_when_cached(tmp_path, monkeypatch):
    """Loader must not re-download when extracted directory already exists."""
    download_called = []

    def fake_download(self):
        download_called.append(True)

    monkeypatch.setattr(UCIHARDataLoader, "_download", fake_download)

    # Use real config — extract path already exists from prior download
    loader = UCIHARDataLoader(Config())
    loader.load()
    assert not download_called, "_download should not be called when cache exists"
