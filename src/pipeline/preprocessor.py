import logging

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


class SignalPreprocessor:
    """Butterworth low-pass filter + z-score normalization + optional EMA smoothing.

    Follows a scikit-learn-style fit/transform interface to prevent data leakage:
    statistics are computed on the training set only and applied to any split.
    """

    def __init__(
        self,
        lowpass_cutoff_hz: float = 20.0,
        sample_rate_hz: int = 50,
        filter_order: int = 4,
        apply_ema: bool = False,
        ema_alpha: float = 0.3,
    ):
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.sample_rate_hz = sample_rate_hz
        self.filter_order = filter_order
        self.apply_ema = apply_ema
        self.ema_alpha = ema_alpha

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._sos = self._build_filter()

    def _build_filter(self) -> np.ndarray:
        nyq = self.sample_rate_hz / 2.0
        normalized_cutoff = self.lowpass_cutoff_hz / nyq
        sos = butter(self.filter_order, normalized_cutoff, btype="low", output="sos")
        return sos

    def _apply_butterworth(self, X: np.ndarray) -> np.ndarray:
        """Apply low-pass Butterworth filter along the time axis (axis=1)."""
        # X shape: (N, timesteps, channels)
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            for c in range(X.shape[2]):
                out[i, :, c] = sosfilt(self._sos, X[i, :, c])
        return out

    def _apply_ema(self, X: np.ndarray) -> np.ndarray:
        """Exponential moving average smoothing along the time axis."""
        alpha = self.ema_alpha
        out = np.empty_like(X)
        out[:, 0, :] = X[:, 0, :]
        for t in range(1, X.shape[1]):
            out[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * out[:, t - 1, :]
        return out

    def fit(self, X: np.ndarray) -> "SignalPreprocessor":
        """Compute per-channel mean and std from training data (after filtering)."""
        X_filtered = self._apply_butterworth(X)
        # Mean/std over samples and timesteps, per channel → shape (9,)
        self._mean = X_filtered.mean(axis=(0, 1))
        self._std = X_filtered.std(axis=(0, 1))
        self._std = np.where(self._std == 0, 1.0, self._std)  # avoid division by zero
        logger.info("SignalPreprocessor fitted: mean=%s std=%s", self._mean, self._std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Filter, z-score normalise, and optionally apply EMA."""
        if self._mean is None or self._std is None:
            raise RuntimeError("SignalPreprocessor must be fitted before transform.")
        X_out = self._apply_butterworth(X)
        X_out = (X_out - self._mean) / self._std
        if self.apply_ema:
            X_out = self._apply_ema(X_out)
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return the transformed result."""
        self.fit(X)
        return self.transform(X)
