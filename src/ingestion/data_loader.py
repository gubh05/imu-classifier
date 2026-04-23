import logging
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from src.utils.config import Config

logger = logging.getLogger(__name__)

ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


class UCIHARDataLoader:
    """Downloads, caches, and parses the UCI HAR raw inertial signal dataset."""

    SIGNAL_NAMES = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.zip_path = config.data_dir / "UCI_HAR_Dataset.zip"
        self.extract_path = config.data_dir / "UCI HAR Dataset"

    def _download(self) -> None:
        logger.info("Downloading UCI HAR Dataset from %s", ZIP_URL)
        response = requests.get(ZIP_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="UCI HAR Dataset"
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        logger.info("Download complete: %s", self.zip_path)

    def _extract(self) -> None:
        logger.info("Extracting %s", self.zip_path)
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.config.data_dir)
        logger.info("Extracted to %s", self.config.data_dir)

    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Load raw inertial signals and labels for a given split (train/test)."""
        signals_dir = self.extract_path / split / "Inertial Signals"
        channels = []
        for name in self.SIGNAL_NAMES:
            filepath = signals_dir / f"{name}_{split}.txt"
            data = np.loadtxt(filepath)  # shape: (N, 128)
            channels.append(data)
        X = np.stack(channels, axis=-1)  # shape: (N, 128, 9)

        label_path = self.extract_path / split / f"y_{split}.txt"
        y = np.loadtxt(label_path, dtype=int) - 1  # 0-indexed

        return X, y

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Download if needed, parse, validate, and return train/test arrays.

        Returns
        -------
        X_train : np.ndarray, shape (7352, 128, 9)
        X_test  : np.ndarray, shape (2947, 128, 9)
        y_train : np.ndarray, shape (7352,)
        y_test  : np.ndarray, shape (2947,)
        """
        if not self.extract_path.exists():
            if not self.zip_path.exists():
                self._download()
            self._extract()
        else:
            logger.info("Using cached dataset at %s", self.extract_path)

        print("Loading train split...")
        X_train, y_train = self._load_split("train")
        print("Loading test split...")
        X_test, y_test = self._load_split("test")

        self._validate(X_train, X_test, y_train, y_test)
        self._print_summary(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test

    def _validate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        if X_train.shape != (7352, 128, 9):
            raise ValueError(f"X_train shape mismatch: expected (7352, 128, 9), got {X_train.shape}")
        if X_test.shape != (2947, 128, 9):
            raise ValueError(f"X_test shape mismatch: expected (2947, 128, 9), got {X_test.shape}")
        if y_train.min() < 0 or y_train.max() > 5:
            raise ValueError(f"y_train labels out of range [0, 5]: min={y_train.min()}, max={y_train.max()}")
        if y_test.min() < 0 or y_test.max() > 5:
            raise ValueError(f"y_test labels out of range [0, 5]: min={y_test.min()}, max={y_test.max()}")
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("NaN values detected in signal data")

    def _print_summary(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        labels = self.config.activity_labels
        print(f"\n--- UCI HAR Dataset Summary ---")
        print(f"Sampling rate : {self.config.sample_rate_hz} Hz")
        print(f"Train samples : {X_train.shape[0]}  shape={X_train.shape}")
        print(f"Test  samples : {X_test.shape[0]}  shape={X_test.shape}")
        print(f"Channels      : {X_train.shape[2]}  (acc_xyz, gyro_xyz, total_acc_xyz)")
        print(f"\nTrain class distribution:")
        for cls, name in labels.items():
            count = (y_train == cls).sum()
            print(f"  {cls}: {name:<22} {count} samples")
        print()
