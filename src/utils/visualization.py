import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

ACTIVITY_LABELS = {
    0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
    3: "SITTING", 4: "STANDING", 5: "LAYING",
}

CHANNEL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_raw_signal(
    window: np.ndarray,
    label: str,
    save_path: Path,
    sample_rate_hz: int = 50,
) -> None:
    """Plot all 9 IMU channels for a single window and save to PNG.

    Parameters
    ----------
    window       : np.ndarray, shape (timesteps, 9)
    label        : Activity name string (for title)
    save_path    : Path to output PNG
    sample_rate_hz : Sampling rate for x-axis in seconds
    """
    _ensure_dir(save_path)
    T = window.shape[0]
    t = np.arange(T) / sample_rate_hz

    fig, axes = plt.subplots(9, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Raw IMU Signal — {label}", fontsize=14, fontweight="bold")

    for i, (ax, ch) in enumerate(zip(axes, CHANNEL_NAMES)):
        ax.plot(t, window[:, i], linewidth=0.8)
        ax.set_ylabel(ch, fontsize=7, rotation=0, labelpad=80, va="center")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved raw signal plot → %s", save_path)


def plot_filtered_overlay(
    raw: np.ndarray,
    filtered: np.ndarray,
    channel_idx: int,
    save_path: Path,
    sample_rate_hz: int = 50,
) -> None:
    """Overlay raw vs Butterworth-filtered signal for one channel.

    Parameters
    ----------
    raw          : np.ndarray, shape (timesteps,)
    filtered     : np.ndarray, shape (timesteps,)
    channel_idx  : Index into CHANNEL_NAMES for the title
    save_path    : Path to output PNG
    """
    _ensure_dir(save_path)
    T = len(raw)
    t = np.arange(T) / sample_rate_hz
    ch_name = CHANNEL_NAMES[channel_idx] if channel_idx < len(CHANNEL_NAMES) else f"ch{channel_idx}"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw, alpha=0.6, linewidth=0.8, label="Raw", color="steelblue")
    ax.plot(t, filtered, linewidth=1.2, label="Butterworth filtered", color="crimson")
    ax.set_title(f"Raw vs. Filtered — {ch_name}", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved filtered overlay plot → %s", save_path)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-N Random Forest feature importances.

    Parameters
    ----------
    importances   : np.ndarray, shape (num_features,)
    feature_names : List of feature name strings
    save_path     : Path to output PNG
    top_n         : Number of top features to display
    """
    _ensure_dir(save_path)
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(top_n), top_vals[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Random Forest Feature Importances", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot → %s", save_path)


def plot_confusion_matrices(
    cms: dict[str, np.ndarray],
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """Side-by-side confusion matrices for multiple models.

    Parameters
    ----------
    cms         : dict mapping model_name → confusion_matrix (np.ndarray)
    save_path   : Path to output PNG
    class_names : Optional list of class label strings
    """
    _ensure_dir(save_path)
    if class_names is None:
        class_names = list(ACTIVITY_LABELS.values())

    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cms.items()):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            linewidths=0.5,
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrices → %s", save_path)


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """Loss and accuracy training curves for CNN and LSTM.

    Parameters
    ----------
    histories : dict mapping model_name → {'loss': [...], 'acc': [...]}
    save_path : Path to output PNG
    """
    _ensure_dir(save_path)
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for row, (model_name, hist) in zip(axes, histories.items()):
        ax_loss, ax_acc = row
        epochs = range(1, len(hist["loss"]) + 1)

        ax_loss.plot(epochs, hist["loss"], marker="o", markersize=3, linewidth=1.2)
        ax_loss.set_title(f"{model_name} — Training Loss", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)

        ax_acc.plot(epochs, hist["acc"], marker="o", markersize=3, linewidth=1.2, color="green")
        ax_acc.set_title(f"{model_name} — Training Accuracy", fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves → %s", save_path)
