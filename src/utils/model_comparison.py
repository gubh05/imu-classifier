"""Aggregate experiment logs and benchmark results to generate comparison plots."""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

RUNS_DIR = Path("outputs/runs")
BENCHMARKS_DIR = Path("outputs/benchmarks")
COMPARISON_DIR = Path("outputs/comparison")


def _load_experiment_metrics() -> dict[str, dict]:
    """Return {model_name: final_metrics} from the most recent run per model.

    Reads run_config.json to identify the model name, then reads metrics.json
    for the final accuracy and f1_macro values.
    """
    if not RUNS_DIR.exists():
        logger.warning("Runs directory not found: %s", RUNS_DIR)
        return {}

    # Collect all runs that have both config and metrics
    runs: list[tuple[str, str, dict]] = []  # (run_id, model_name, metrics)
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "run_config.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            with open(metrics_path) as f:
                metrics = json.load(f)
            run_id = cfg.get("run_id", run_dir.name)
            model_name = cfg.get("run_name", "unknown")
            final = metrics.get("final", {})
            if "accuracy" in final and "f1_macro" in final:
                runs.append((run_id, model_name, final))
        except Exception as exc:
            logger.warning("Skipping run %s: %s", run_dir.name, exc)

    # Keep only the most recent run per model (run_id is timestamp-prefixed)
    best: dict[str, tuple[str, dict]] = {}
    for run_id, model_name, metrics in runs:
        if model_name not in best or run_id > best[model_name][0]:
            best[model_name] = (run_id, metrics)

    return {name: data[1] for name, data in best.items()}


def _load_benchmark_metrics() -> dict[str, dict]:
    """Return {model_key: benchmark_result} from outputs/benchmarks/*.json."""
    if not BENCHMARKS_DIR.exists():
        logger.warning("Benchmarks directory not found: %s", BENCHMARKS_DIR)
        return {}

    results: dict[str, dict] = {}
    for path in BENCHMARKS_DIR.glob("*_latency.json"):
        model_key = path.stem.replace("_latency", "")
        try:
            with open(path) as f:
                results[model_key] = json.load(f)
        except Exception as exc:
            logger.warning("Skipping benchmark %s: %s", path.name, exc)
    return results


def _bar_chart(
    values: dict[str, float],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    """Render and save a bar chart."""
    models = list(values.keys())
    scores = [values[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    bars = ax.bar(x, scores, width=0.5, color=["steelblue", "darkorange", "seagreen"][: len(models)])
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def _scatter_latency_vs_accuracy(
    exp_metrics: dict[str, dict],
    bench_metrics: dict[str, dict],
    save_path: Path,
) -> None:
    """Scatter plot of mean latency (ms) vs accuracy for each model."""
    # Find models present in both datasets
    models = sorted(set(exp_metrics) & set(bench_metrics))
    if not models:
        logger.warning("No overlapping models between experiment and benchmark data.")
        return

    latencies = [bench_metrics[m]["mean_latency_ms"] for m in models]
    accuracies = [exp_metrics[m]["accuracy"] for m in models]
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model, lat, acc) in enumerate(zip(models, latencies, accuracies)):
        ax.scatter(lat, acc, s=120, color=colors[i % len(colors)], zorder=3, label=model.upper())
        ax.annotate(
            model.upper(),
            (lat, acc),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Latency vs Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def generate_all_plots() -> None:
    """Aggregate metrics and generate all comparison plots.

    Reads from:
      - outputs/runs/*/metrics.json  (experiment logs)
      - outputs/benchmarks/*.json    (latency benchmarks)

    Writes to:
      - outputs/comparison/accuracy_comparison.png
      - outputs/comparison/f1_comparison.png
      - outputs/comparison/latency_vs_accuracy.png
    """
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    exp_metrics = _load_experiment_metrics()
    bench_metrics = _load_benchmark_metrics()

    if not exp_metrics:
        logger.warning("No experiment metrics found — skipping accuracy/F1 plots.")
    else:
        accuracy_values = {m: d["accuracy"] for m, d in exp_metrics.items()}
        _bar_chart(
            accuracy_values,
            title="Model Accuracy Comparison",
            ylabel="Accuracy",
            save_path=COMPARISON_DIR / "accuracy_comparison.png",
        )

        f1_values = {m: d["f1_macro"] for m, d in exp_metrics.items()}
        _bar_chart(
            f1_values,
            title="Model F1 Score Comparison (Macro)",
            ylabel="F1 Score (Macro)",
            save_path=COMPARISON_DIR / "f1_comparison.png",
        )

    if not exp_metrics or not bench_metrics:
        logger.warning("Missing experiment or benchmark data — skipping latency vs accuracy plot.")
    else:
        _scatter_latency_vs_accuracy(
            exp_metrics,
            bench_metrics,
            save_path=COMPARISON_DIR / "latency_vs_accuracy.png",
        )

    generated = list(COMPARISON_DIR.glob("*.png"))
    print(f"Generated {len(generated)} plot(s) in {COMPARISON_DIR}:")
    for p in sorted(generated):
        print(f"  {p.name}")
