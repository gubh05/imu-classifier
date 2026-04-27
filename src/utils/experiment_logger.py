"""Lightweight, file-based experiment tracker for training runs."""
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python before JSON serialisation."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _cast_to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _cast_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Git / environment helpers
# ---------------------------------------------------------------------------

def _collect_git_info() -> dict:
    """Return git commit/branch/dirty status. Falls back to 'unavailable' on any error."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return {"commit_hash": commit, "branch": branch, "dirty": bool(status)}
    except Exception as exc:
        log.warning("ExperimentLogger: git info unavailable — %s", exc)
        return {"commit_hash": "unavailable", "branch": "unavailable", "dirty": "unavailable"}


def _collect_env_info() -> dict:
    """Return Python / platform / torch / numpy versions."""
    info: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import torch  # optional
        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "unavailable"
    return info


def _config_to_dict(config: Any) -> dict:
    """Serialise a Config dataclass to a plain dict, skipping non-scalar fields."""
    try:
        raw = asdict(config)
    except TypeError:
        raw = vars(config) if hasattr(config, "__dict__") else {}
    result: dict = {}
    for k, v in raw.items():
        if isinstance(v, Path):
            result[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            result[k] = v
        # skip dict/list fields (e.g. activity_labels)
    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}m {secs}s"


def _build_report(
    run_config: dict,
    epoch_log: list,
    final_metrics: dict,
    config: Any,
    duration: float,
) -> str:
    """Produce a human-readable Markdown run report."""
    run_id = run_config["run_id"]
    run_name = run_config["run_name"]
    started_at = run_config["started_at"].replace("T", " ").replace("Z", " UTC")

    # Infer model label from run_name
    name_upper = run_name.upper()
    if name_upper in ("CNN", "LSTM", "RF"):
        model_label = name_upper
    else:
        model_label = run_name.split("_")[0].upper()

    cfg = run_config.get("config", {})

    # Pick the right epoch count
    if model_label == "CNN":
        epochs = cfg.get("cnn_epochs", "-")
    elif model_label == "LSTM":
        epochs = cfg.get("lstm_epochs", "-")
    else:
        epochs = "-"

    acc = final_metrics.get("accuracy", 0.0)
    f1_macro = final_metrics.get("f1_macro", final_metrics.get("f1", 0.0))

    lines = [
        "# Experiment Run Report",
        f"**Run ID:** {run_id}",
        f"**Date:** {started_at}",
        f"**Duration:** {_fmt_duration(duration)}",
        "",
        "## Configuration",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | {model_label} |",
        f"| Epochs | {epochs} |",
        f"| Batch Size | {cfg.get('batch_size', '-')} |",
        f"| Learning Rate | {cfg.get('learning_rate', '-')} |",
        f"| Random Seed | {cfg.get('random_seed', '-')} |",
        "",
        "## Results",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Test Accuracy | {acc:.1%} |",
        f"| F1 Score (macro) | {f1_macro:.1%} |",
    ]

    # Per-class F1
    f1_per_class = final_metrics.get("f1_per_class")
    if f1_per_class:
        activity_labels: dict = getattr(config, "activity_labels", {})
        lines += [
            "",
            "## Per-Class F1 Scores",
            "| Activity | F1 |",
            "|----------|----|",
        ]
        for i, f1v in enumerate(f1_per_class):
            label = activity_labels.get(i, str(i))
            lines.append(f"| {label} | {f1v:.4f} |")

    # Training curve summary
    if epoch_log:
        train_losses = [e["train_loss"] for e in epoch_log]
        val_losses = [e["val_loss"] for e in epoch_log if "val_loss" in e]
        lines += ["", "## Training Curve Summary"]
        if val_losses:
            best_val = min(val_losses)
            best_epoch = next(
                e["epoch"] for e in epoch_log if e.get("val_loss") == best_val
            )
            lines.append(f"- Best validation loss: {best_val:.4f} (epoch {best_epoch})")
        lines.append(f"- Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            lines.append(f"- Final val loss: {val_losses[-1]:.4f}")

    # Saved artefacts
    artefacts = run_config.get("artefacts", [])
    if artefacts:
        lines += ["", "## Saved Artefacts"]
        for p in artefacts:
            lines.append(f"- `{p}`")

    # Environment
    env = run_config.get("environment", {})
    git = run_config.get("git", {})
    lines += [
        "",
        "## Environment",
        f"- Python: {env.get('python_version', '-')}",
        f"- PyTorch: {env.get('torch_version', '-')}",
        f"- NumPy: {env.get('numpy_version', '-')}",
        f"- Git commit: {git.get('commit_hash', '-')} ({git.get('branch', '-')})",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Lightweight, file-based experiment tracker.

    Creates a timestamped run directory under outputs/runs/ and writes:
      - run_config.json   : hyperparameters, environment, git info
      - metrics.json      : per-epoch losses + final evaluation scores
      - run_report.md     : human-readable summary of the run
    """

    def __init__(self, run_name: str, config: Any, base_dir: Path = Path("outputs/runs")):
        """
        Args:
            run_name:  Short label for the run, e.g. "cnn_lr0.001_e30".
            config:    The project Config dataclass instance.
            base_dir:  Root directory where run folders are created.
        """
        self.run_name = run_name
        self._config = config

        now = datetime.now(timezone.utc)
        self.run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{run_name}"
        self._started_at: datetime = now
        self._start_time: float = time.time()

        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._epoch_log: list[dict] = []
        self._final_metrics: dict = {}
        self._artefact_paths: list[str] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append one epoch's metrics. Call once per epoch during training.

        Args:
            epoch:      1-based epoch number.
            train_loss: Average training loss for this epoch.
            val_loss:   Validation loss (optional).
            extra:      Additional metrics, e.g. {"train_acc": 0.91}.
        """
        try:
            entry: dict = {"epoch": int(epoch), "train_loss": float(train_loss)}
            if val_loss is not None:
                entry["val_loss"] = float(val_loss)
            if extra:
                entry.update(_cast_to_native(extra))
            self._epoch_log.append(entry)
        except Exception as exc:
            log.warning("ExperimentLogger.log_epoch failed: %s", exc)

    def log_final_metrics(self, metrics: dict) -> None:
        """Record end-of-training evaluation metrics.

        Args:
            metrics: dict with keys such as accuracy, f1_macro, f1_per_class (list),
                     confusion_matrix (list of lists). All numpy types are cast to
                     native Python types before storage.
        """
        try:
            self._final_metrics = _cast_to_native(metrics)
        except Exception as exc:
            log.warning("ExperimentLogger.log_final_metrics failed: %s", exc)

    def log_artefact(self, path: "str | Path") -> None:
        """Record the path of a saved artefact (model checkpoint, plot).

        Paths are stored relative to the project root.
        """
        try:
            self._artefact_paths.append(str(path))
        except Exception as exc:
            log.warning("ExperimentLogger.log_artefact failed: %s", exc)

    def finish(self) -> Path:
        """Finalise the run. Writes run_config.json, metrics.json, run_report.md.

        Returns:
            The run directory path.
        """
        try:
            finished_at = datetime.now(timezone.utc)
            duration = time.time() - self._start_time

            git_info = _collect_git_info()
            env_info = _collect_env_info()
            config_dict = _config_to_dict(self._config)

            run_config: dict = {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "started_at": self._started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_seconds": round(duration, 1),
                "config": config_dict,
                "environment": env_info,
                "git": git_info,
                "artefacts": self._artefact_paths,
            }
            (self.run_dir / "run_config.json").write_text(
                json.dumps(run_config, indent=2, cls=_NumpyEncoder),
                encoding="utf-8",
            )

            metrics_out: dict = {
                "epoch_log": self._epoch_log,
                "final": self._final_metrics,
            }
            (self.run_dir / "metrics.json").write_text(
                json.dumps(metrics_out, indent=2, cls=_NumpyEncoder),
                encoding="utf-8",
            )

            report = _build_report(
                run_config=run_config,
                epoch_log=self._epoch_log,
                final_metrics=self._final_metrics,
                config=self._config,
                duration=duration,
            )
            (self.run_dir / "run_report.md").write_text(report, encoding="utf-8")

            log.info(
                "ExperimentLogger: run '%s' complete — acc=%.4f  dir=%s",
                self.run_id,
                self._final_metrics.get("accuracy", 0.0),
                self.run_dir,
            )
        except Exception as exc:
            log.warning("ExperimentLogger.finish failed: %s", exc)

        return self.run_dir
