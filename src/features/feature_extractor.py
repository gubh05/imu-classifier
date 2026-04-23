import logging

import numpy as np
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)

# Channel names (in order they are stacked in X)
_CHANNEL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# Time-domain feature names (per channel)
_TIME_FEATURES = [
    "mean", "std", "min", "max", "rms",
    "peak_to_peak", "zero_crossing_rate", "energy",
]

# Frequency-domain feature names (per channel)
_FREQ_FEATURES = [
    "fft_dominant_freq", "fft_spectral_entropy",
    "power_low", "power_mid", "power_high",
]

_FEATURES_PER_CHANNEL = _TIME_FEATURES + _FREQ_FEATURES  # 13 features


class FeatureExtractor:
    """Extracts time- and frequency-domain features from IMU windows.

    Input:  X of shape (N, timesteps, channels)
    Output: feature matrix of shape (N, num_features)
            where num_features = n_channels * 13 = 9 * 13 = 117
    """

    def __init__(self, sample_rate_hz: int = 50):
        self.sample_rate_hz = sample_rate_hz

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(w: np.ndarray) -> float:
        return float(np.mean(w))

    @staticmethod
    def _std(w: np.ndarray) -> float:
        return float(np.std(w))

    @staticmethod
    def _min(w: np.ndarray) -> float:
        return float(np.min(w))

    @staticmethod
    def _max(w: np.ndarray) -> float:
        return float(np.max(w))

    @staticmethod
    def _rms(w: np.ndarray) -> float:
        return float(np.sqrt(np.mean(w ** 2)))

    @staticmethod
    def _peak_to_peak(w: np.ndarray) -> float:
        return float(np.max(w) - np.min(w))

    @staticmethod
    def _zero_crossing_rate(w: np.ndarray) -> float:
        signs = np.sign(w)
        signs[signs == 0] = 1  # treat zero as positive to avoid ambiguity
        crossings = np.sum(np.diff(signs) != 0)
        return float(crossings / len(w))

    @staticmethod
    def _energy(w: np.ndarray) -> float:
        return float(np.sum(w ** 2))

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------

    def _freq_features(self, w: np.ndarray) -> list[float]:
        n = len(w)
        spectrum = np.abs(rfft(w)) ** 2  # power spectrum
        freqs = rfftfreq(n, d=1.0 / self.sample_rate_hz)

        # Dominant frequency (bin with max power, excluding DC)
        if len(spectrum) > 1:
            dominant_idx = np.argmax(spectrum[1:]) + 1
        else:
            dominant_idx = 0
        dominant_freq = float(freqs[dominant_idx])

        # Spectral entropy
        power_sum = spectrum.sum()
        if power_sum > 0:
            p_norm = spectrum / power_sum
            # Avoid log(0) by clipping
            p_norm = np.clip(p_norm, 1e-12, None)
            spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm)))
        else:
            spectral_entropy = 0.0

        # Band power
        low_mask = freqs <= 5.0
        mid_mask = (freqs > 5.0) & (freqs <= 15.0)
        high_mask = (freqs > 15.0) & (freqs <= 25.0)
        power_low = float(spectrum[low_mask].sum())
        power_mid = float(spectrum[mid_mask].sum())
        power_high = float(spectrum[high_mask].sum())

        return [dominant_freq, spectral_entropy, power_low, power_mid, power_high]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_window_channel(self, w: np.ndarray) -> list[float]:
        """Extract all features for a single (timesteps,) channel window."""
        time_feats = [
            self._mean(w),
            self._std(w),
            self._min(w),
            self._max(w),
            self._rms(w),
            self._peak_to_peak(w),
            self._zero_crossing_rate(w),
            self._energy(w),
        ]
        freq_feats = self._freq_features(w)
        return time_feats + freq_feats

    def extract(self, X: np.ndarray) -> np.ndarray:
        """Extract features from all windows.

        Parameters
        ----------
        X : np.ndarray, shape (N, timesteps, channels)

        Returns
        -------
        np.ndarray, shape (N, n_channels * features_per_channel)
        """
        N, T, C = X.shape
        n_feats = C * len(_FEATURES_PER_CHANNEL)
        out = np.empty((N, n_feats), dtype=np.float32)

        for i in range(N):
            row: list[float] = []
            for c in range(C):
                row.extend(self._extract_window_channel(X[i, :, c].astype(np.float64)))
            out[i] = row

        logger.debug("Extracted features: shape=%s", out.shape)
        return out

    def feature_names(self) -> list[str]:
        """Return a list of feature names matching the column order of extract()."""
        names: list[str] = []
        for ch in _CHANNEL_NAMES:
            for feat in _FEATURES_PER_CHANNEL:
                names.append(f"{ch}__{feat}")
        return names
