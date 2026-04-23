"""End-to-end training script: loads UCI HAR data, trains RF / CNN / LSTM, saves results."""
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from src.features.feature_extractor import FeatureExtractor
from src.ingestion.data_loader import UCIHARDataLoader
from src.models.cnn_classifier import CNNClassifier
from src.models.lstm_classifier import LSTMClassifier
from src.models.rf_baseline import RFClassifier
from src.pipeline.preprocessor import SignalPreprocessor
from src.pipeline.sensor_pipeline import SensorPipeline
from src.utils.config import Config
from src.utils.visualization import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_filtered_overlay,
    plot_raw_signal,
    plot_training_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    t0 = time.time()
    rf_pipe = SensorPipeline(preprocessor, extractor, RFClassifier())
    # preprocessor already fitted — fit() re-uses it via fit_transform (re-fit is harmless
    # but to be safe we train only the model on already-processed features)
    rf_model = RFClassifier()
    rf_model.fit(X_train_feat, y_train)
    rf_elapsed = time.time() - t0

    from sklearn.metrics import accuracy_score, f1_score
    rf_preds = rf_model.predict(X_test_feat)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted")
    from sklearn.metrics import confusion_matrix
    confusion_matrices["RF"] = confusion_matrix(y_test, rf_preds)
    results["RF"] = {"accuracy": rf_acc, "f1": rf_f1, "time_s": rf_elapsed}

    # Save RF model
    joblib.dump(rf_model, cfg.output_dir / "rf_model.pkl")
    print(f"      RF done in {rf_elapsed:.1f}s — acc={rf_acc:.4f}  f1={rf_f1:.4f}")

    # Feature importance plot (RF only)
    plot_feature_importance(
        rf_model.feature_importances_,
        extractor.feature_names(),
        save_path=plots_dir / "feature_importance.png",
    )

    # --- CNN ---
    print("\n[5/6] Training CNN …")
    t0 = time.time()
    cnn_model = CNNClassifier(
        epochs=cfg.cnn_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
    )
    cnn_model.fit(X_train_pp, y_train, X_val=X_test_pp, y_val=y_test)
    cnn_elapsed = time.time() - t0

    cnn_preds = cnn_model.predict(X_test_pp)
    cnn_acc = accuracy_score(y_test, cnn_preds)
    cnn_f1 = f1_score(y_test, cnn_preds, average="weighted")
    confusion_matrices["CNN"] = confusion_matrix(y_test, cnn_preds)
    results["CNN"] = {"accuracy": cnn_acc, "f1": cnn_f1, "time_s": cnn_elapsed}
    training_histories["CNN"] = {"loss": cnn_model.train_losses, "acc": cnn_model.train_accs}

    torch.save(cnn_model.model.state_dict(), cfg.output_dir / "cnn_best.pt")
    print(f"      CNN done in {cnn_elapsed:.1f}s — acc={cnn_acc:.4f}  f1={cnn_f1:.4f}")

    # --- LSTM ---
    print("\n[6/6] Training LSTM …")
    t0 = time.time()
    lstm_model = LSTMClassifier(
        epochs=cfg.lstm_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
    )
    lstm_model.fit(X_train_pp, y_train, X_val=X_test_pp, y_val=y_test)
    lstm_elapsed = time.time() - t0

    lstm_preds = lstm_model.predict(X_test_pp)
    lstm_acc = accuracy_score(y_test, lstm_preds)
    lstm_f1 = f1_score(y_test, lstm_preds, average="weighted")
    confusion_matrices["LSTM"] = confusion_matrix(y_test, lstm_preds)
    results["LSTM"] = {"accuracy": lstm_acc, "f1": lstm_f1, "time_s": lstm_elapsed}
    training_histories["LSTM"] = {"loss": lstm_model.train_losses, "acc": lstm_model.train_accs}

    torch.save(lstm_model.model.state_dict(), cfg.output_dir / "lstm_best.pt")
    print(f"      LSTM done in {lstm_elapsed:.1f}s — acc={lstm_acc:.4f}  f1={lstm_f1:.4f}")

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
