"""CLI tool to run latency benchmarks for a trained model pipeline.

Usage
-----
    python scripts/benchmark_latency.py --model rf
    python scripts/benchmark_latency.py --model cnn
    python scripts/benchmark_latency.py --model lstm
"""
import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.api.app import _load_pipeline
from src.benchmarking.latency_benchmark import LatencyBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model inference latency.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["rf", "cnn", "lstm"],
        help="Model to benchmark.",
    )
    args = parser.parse_args()

    print(f"Loading pipeline: {args.model} ...")
    pipeline = _load_pipeline(args.model)

    # Generate synthetic input — shape (1, 128, 9)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1, 128, 9)).astype(np.float32)

    print(f"Running benchmark (warmup=10, runs=100) ...")
    bench = LatencyBenchmark(pipeline)
    result = bench.run(X)
    out_path = bench.save(result, args.model)

    print(f"\nResults for model: {args.model}")
    print(f"  Mean latency   : {result['mean_latency_ms']:.3f} ms")
    print(f"  Std deviation  : {result['std_latency_ms']:.3f} ms")
    print(f"  P95 latency    : {result['p95_latency_ms']:.3f} ms")
    print(f"  Throughput     : {result['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
