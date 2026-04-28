"""Latency benchmarking for trained SensorPipeline instances."""
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs/benchmarks")


class LatencyBenchmark:
    """Measure single-sample inference latency of a SensorPipeline.

    Parameters
    ----------
    pipeline:
        A loaded SensorPipeline instance.
    warmup_runs:
        Number of un-timed warm-up predictions before measurement.
    benchmark_runs:
        Number of timed predictions used to compute statistics.
    """

    def __init__(self, pipeline, warmup_runs: int = 10, benchmark_runs: int = 100):
        self.pipeline = pipeline
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self._model_name: str = type(pipeline.model).__name__

    def run(self, X: np.ndarray) -> dict:
        """Benchmark the pipeline on a single sample drawn from X.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (N, 128, 9). One sample is selected per inference call.

        Returns
        -------
        dict
            Benchmark metrics including mean/std/p95 latency and throughput.
        """
        # Use the first sample, keep batch dimension: (1, 128, 9)
        sample = X[:1]

        # Pre-compute the prepared input to avoid repeated preprocessing overhead
        # in the timing loop (we time the model inference step only, matching
        # the API's latency measurement which uses _prepare_X + predict_proba).
        pipeline = self.pipeline

        def _infer() -> None:
            X_prepared = pipeline._prepare_X(sample)
            pipeline.model.predict_proba(X_prepared)

        # Warm-up
        for _ in range(self.warmup_runs):
            _infer()

        # Timed runs
        latencies_ms: list[float] = []
        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            _infer()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(latencies_ms)
        mean_ms = float(np.mean(arr))
        std_ms = float(np.std(arr))
        p95_ms = float(np.percentile(arr, 95))
        throughput = 1000.0 / mean_ms  # samples per second

        result = {
            "model": self._model_name,
            "mean_latency_ms": round(mean_ms, 3),
            "std_latency_ms": round(std_ms, 3),
            "p95_latency_ms": round(p95_ms, 3),
            "throughput_samples_per_sec": round(throughput, 3),
        }
        return result

    def save(self, result: dict, model_key: str) -> Path:
        """Write benchmark result to outputs/benchmarks/<model_key>_latency.json."""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / f"{model_key}_latency.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Benchmark saved to %s", out_path)
        return out_path
