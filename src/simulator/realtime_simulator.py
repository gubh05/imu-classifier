import logging
from typing import Generator

import numpy as np

from src.pipeline.sensor_pipeline import SensorPipeline
from src.utils.config import Config

logger = logging.getLogger(__name__)


class RealTimeSimulator:
    """Simulates real-time IMU stream inference using a sliding window.

    Slides a window of `window_size` samples across `raw_signal` with a step
    of `step_size`, runs the SensorPipeline on each window, and yields
    (step_index, predicted_label, confidence) for every window.

    Parameters
    ----------
    pipeline    : Fitted SensorPipeline
    window_size : Number of timesteps per window (default 128 = 2.56 s at 50 Hz)
    step_size   : Slide increment (default 32 = 0.64 s update rate)
    config      : Config instance for label names
    """

    def __init__(
        self,
        pipeline: SensorPipeline,
        window_size: int = 128,
        step_size: int = 32,
        config: Config | None = None,
    ):
        self.pipeline = pipeline
        self.window_size = window_size
        self.step_size = step_size
        self.config = config or Config()

    def stream(
        self, raw_signal: np.ndarray
    ) -> Generator[tuple[int, str, float], None, None]:
        """Slide across raw_signal and yield predictions for each window.

        Parameters
        ----------
        raw_signal : np.ndarray, shape (T, 9)
            Continuous IMU signal — T timesteps, 9 channels.

        Yields
        ------
        (step_index, predicted_label_name, confidence_pct)
        """
        T, C = raw_signal.shape
        if C != 9:
            raise ValueError(f"raw_signal must have 9 channels, got {C}")
        if T < self.window_size:
            raise ValueError(
                f"Signal length {T} is shorter than window_size {self.window_size}"
            )

        n_windows = (T - self.window_size) // self.step_size + 1
        logger.info(
            "RealTimeSimulator: signal_len=%d, window=%d, step=%d → %d windows",
            T, self.window_size, self.step_size, n_windows,
        )

        print(
            f"\n{'Step':>6}  {'Time (s)':>10}  {'Activity':<22}  {'Confidence':>10}"
        )
        print("-" * 56)

        for i in range(n_windows):
            start = i * self.step_size
            end = start + self.window_size
            window = raw_signal[start:end]                    # (window_size, 9)
            X_window = window[np.newaxis, :, :]               # (1, window_size, 9)

            # predict returns shape (1,) — grab scalar
            label_idx = int(self.pipeline.predict(X_window)[0])
            label_name = self.config.activity_labels.get(label_idx, str(label_idx))

            # Confidence from predict_proba if the model supports it
            if hasattr(self.pipeline.model, "predict_proba"):
                X_prepared = self.pipeline._prepare_X(X_window)
                proba = self.pipeline.model.predict_proba(X_prepared)
                confidence = float(np.max(proba)) * 100.0
            else:
                confidence = 100.0

            time_s = start / self.config.sample_rate_hz
            print(f"{i:>6}  {time_s:>9.2f}s  {label_name:<22}  {confidence:>9.1f}%")

            yield (i, label_name, confidence)
