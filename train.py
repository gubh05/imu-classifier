"""End-to-end training script: loads UCI HAR data, trains RF / CNN / LSTM, saves results."""
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.features.feature_extractor import FeatureExtractor
from src.ingestion.data_loader import UCIHARDataLoader
from src.models.cnn_classifier import CNNClassifier
from src.models.lstm_classifier import LSTMClassifier
from src.models.rf_baseline import RFClassifier
from src.pipeline.preprocessor import SignalPreprocessor
from src.pipeline.sensor_pipeline import SensorPipeline
from src.utils.config import Config
from src.utils.experiment_logger import ExperimentLogger
from src.utils.visualization import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_filtered_overlay,
    plot_raw_signal,
    plot_training_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy, f1_macro, f1_per_class, confusion_matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  IMU Sports Activity Classifier — Training Run")
    print("=" * 60)

    loader = UCIHARDataLoader(cfg)
    X_train, X_test, y_train, y_test = loader.load()

    # ------------------------------------------------------------------
    # 2. Preprocess (fit on train only)
    # ------------------------------------------------------------------
    print("\n[1/6] Preprocessing …")
    preprocessor = SignalPreprocessor()
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    print("[2/6] Extracting features …")
    extractor = FeatureExtractor(sample_rate_hz=cfg.sample_rate_hz)
    X_train_feat = extractor.extract(X_train_pp)
    X_test_feat = extractor.extract(X_test_pp)
    print(f"      Feature matrix: train={X_train_feat.shape}, test={X_test_feat.shape}")

    # ------------------------------------------------------------------
    # 4. Visualize raw signal and filter effect (before model training)
    # ------------------------------------------------------------------
    print("[3/6] Generating signal plots …")
    for cls_idx, cls_name in cfg.activity_labels.items():
        idx = np.where(y_train == cls_idx)[0][0]
        plot_raw_signal(
            X_train[idx],
            label=cls_name,
            save_path=plots_dir / f"raw_signal_{cls_name.lower()}.png",
        )

    plot_filtered_overlay(
        raw=X_train[0, :, 0],
        filtered=X_train_pp[0, :, 0],
        channel_idx=0,
        save_path=plots_dir / "filtered_overlay.png",
    )

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    results = {}
    confusion_matrices = {}
    training_histories = {}

    # --- Random Forest ---
    print("\n[4/6] Training Random Forest …")
    rf_exp = ExperimentLogger(run_name="rf", config=cfg)
    t0 = time.time()
    rf_pipe = SensorPipeline(preprocessor, extractor, RFClassifier())
    rf_model = RFClassifier()
    rf_model.fit(X_train_feat, y_train)
    rf_elapsed = time.time() - t0

    rf_preds = rf_model.predict(X_test_feat)
    rf_metrics = _eval_metrics(y_test, rf_preds)
    rf_acc = rf_metrics["accuracy"]
    rf_f1 = float(f1_score(y_test, rf_preds, average="weighted"))
    confusion_matrices["RF"] = np.array(rf_metrics["confusion_matrix"])
    results["RF"] = {"accuracy": rf_acc, "f1": rf_f1, "time_s": rf_elapsed}

    joblib.dump(rf_model, cfg.output_dir / "rf_model.pkl")
    print(f"      RF done in {rf_elapsed:.1f}s — acc={rf_acc:.4f}  f1={rf_f1:.4f}")

    plot_feature_importance(
        rf_model.feature_importances_,
        extractor.feature_names(),
        save_path=plots_dir / "feature_importance.png",
    )

    rf_exp.log_final_metrics(rf_metrics)
    rf_exp.log_artefact("outputs/rf_model.pkl")
    rf_exp.log_artefact("outputs/plots/feature_importance.png")
    rf_run_dir = rf_exp.finish()
    print(f"RF run saved to: {rf_run_dir}")

    # --- CNN ---
    print("\n[5/6] Training CNN …")
    cnn_exp = ExperimentLogger(run_name="cnn", config=cfg)
    t0 = time.time()
    cnn_model = CNNClassifier(
        epochs=cfg.cnn_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
    )
    cnn_model.fit(X_train_pp, y_train, X_val=X_test_pp, y_val=y_test)
    cnn_elapsed = time.time() - t0

    for epoch_idx, train_loss in enumerate(cnn_model.train_losses):
        cnn_exp.log_epoch(
            epoch_idx + 1,
            train_loss=train_loss,
            extra={"train_acc": cnn_model.train_accs[epoch_idx]},
        )

    cnn_preds = cnn_model.predict(X_test_pp)
    cnn_metrics = _eval_metrics(y_test, cnn_preds)
    cnn_acc = cnn_metrics["accuracy"]
    cnn_f1 = float(f1_score(y_test, cnn_preds, average="weighted"))
    confusion_matrices["CNN"] = np.array(cnn_metrics["confusion_matrix"])
    results["CNN"] = {"accuracy": cnn_acc, "f1": cnn_f1, "time_s": cnn_elapsed}
    training_histories["CNN"] = {"loss": cnn_model.train_losses, "acc": cnn_model.train_accs}

    torch.save(cnn_model.model.state_dict(), cfg.output_dir / "cnn_best.pt")
    print(f"      CNN done in {cnn_elapsed:.1f}s — acc={cnn_acc:.4f}  f1={cnn_f1:.4f}")

    cnn_exp.log_final_metrics(cnn_metrics)
    cnn_exp.log_artefact("outputs/cnn_best.pt")
    cnn_exp.log_artefact("outputs/plots/confusion_matrices.png")
    cnn_run_dir = cnn_exp.finish()
    print(f"CNN run saved to: {cnn_run_dir}")

    # --- LSTM ---
    print("\n[6/6] Training LSTM …")
    lstm_exp = ExperimentLogger(run_name="lstm", config=cfg)
    t0 = time.time()
    lstm_model = LSTMClassifier(
        epochs=cfg.lstm_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
    )
    lstm_model.fit(X_train_pp, y_train, X_val=X_test_pp, y_val=y_test)
    lstm_elapsed = time.time() - t0

    for epoch_idx, train_loss in enumerate(lstm_model.train_losses):
        lstm_exp.log_epoch(
            epoch_idx + 1,
            train_loss=train_loss,
            extra={"train_acc": lstm_model.train_accs[epoch_idx]},
        )

    lstm_preds = lstm_model.predict(X_test_pp)
    lstm_metrics = _eval_metrics(y_test, lstm_preds)
    lstm_acc = lstm_metrics["accuracy"]
    lstm_f1 = float(f1_score(y_test, lstm_preds, average="weighted"))
    confusion_matrices["LSTM"] = np.array(lstm_metrics["confusion_matrix"])
    results["LSTM"] = {"accuracy": lstm_acc, "f1": lstm_f1, "time_s": lstm_elapsed}
    training_histories["LSTM"] = {"loss": lstm_model.train_losses, "acc": lstm_model.train_accs}

    torch.save(lstm_model.model.state_dict(), cfg.output_dir / "lstm_best.pt")
    print(f"      LSTM done in {lstm_elapsed:.1f}s — acc={lstm_acc:.4f}  f1={lstm_f1:.4f}")

    lstm_exp.log_final_metrics(lstm_metrics)
    lstm_exp.log_artefact("outputs/lstm_best.pt")
    lstm_exp.log_artefact("outputs/plots/training_curves.png")
    lstm_run_dir = lstm_exp.finish()
    print(f"LSTM run saved to: {lstm_run_dir}")

    # ------------------------------------------------------------------
    # 6. Save full pipelines, plots, and results table
    # ------------------------------------------------------------------
    rf_full_pipe = SensorPipeline(preprocessor, extractor, rf_model)
    rf_full_pipe.save(str(cfg.output_dir / "pipeline_rf.pkl"))

    plot_confusion_matrices(confusion_matrices, save_path=plots_dir / "confusion_matrices.png")
    plot_training_curves(training_histories, save_path=plots_dir / "training_curves.png")

    # Save results to disk for notebooks
    np.save(cfg.output_dir / "results.npy", results)

    # ------------------------------------------------------------------
    # 7. Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  {'Model':<8}  {'Accuracy':>10}  {'F1 (weighted)':>14}  {'Time':>8}")
    print("-" * 60)
    for model_name, m in results.items():
        print(
            f"  {model_name:<8}  {m['accuracy']:>10.4f}  {m['f1']:>14.4f}  {m['time_s']:>7.1f}s"
        )
    print("=" * 60)
    print(f"\nCheckpoints saved to: {cfg.output_dir}")
    print(f"Plots saved to      : {plots_dir}\n")


if __name__ == "__main__":
    main()
