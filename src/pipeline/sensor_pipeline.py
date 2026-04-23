import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.features.feature_extractor import FeatureExtractor
from src.pipeline.preprocessor import SignalPreprocessor

logger = logging.getLogger(__name__)


class SensorPipeline:
    """Top-level orchestrator: preprocessing → feature extraction → model.

    Works with any model that implements .fit(X, y), .predict(X), and
    optionally .predict_proba(X). Compatible with RFClassifier,
    CNNClassifier, and LSTMClassifier.
    """

    def __init__(
        self,
        preprocessor: SignalPreprocessor,
        extractor: FeatureExtractor,
        model,
    ):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.model = model
        self._model_type = type(model).__name__

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_deep_model(self) -> bool:
        """True for CNN/LSTM models that operate on raw (N, T, C) windows."""
        return self._model_type in ("CNNClassifier", "LSTMClassifier")

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Return the right representation depending on model type.

        - RF  : preprocessed + feature-extracted flat vectors  (N, num_features)
        - CNN/LSTM : preprocessed raw windows                  (N, 128, 9)
        """
        X_pp = self.preprocessor.transform(X)
        if self._is_deep_model():
            return X_pp
        return self.extractor.extract(X_pp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SensorPipeline":
        """Fit preprocessor on training data, then fit the model.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, 128, 9)
        y_train : np.ndarray, shape (N,)
        """
        logger.info("SensorPipeline.fit — model: %s", self._model_type)
        X_pp = self.preprocessor.fit_transform(X_train)

        if self._is_deep_model():
            self.model.fit(X_pp, y_train)
        else:
            X_feat = self.extractor.extract(X_pp)
            self.model.fit(X_feat, y_train)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Preprocess, extract features (if needed), and predict.

        Returns
        -------
        np.ndarray, shape (N,) — integer class labels
        """
        return self.model.predict(self._prepare_X(X))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Run predict and compute accuracy, F1, and confusion matrix.

        Returns
        -------
        dict with keys: 'accuracy', 'f1', 'confusion_matrix'
        """
        y_pred = self.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        cm = confusion_matrix(y_test, y_pred)
        logger.info(
            "SensorPipeline.evaluate — accuracy=%.4f f1=%.4f", acc, f1
        )
        return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}

    def save(self, path: str) -> None:
        """Serialize the entire pipeline (preprocessor + extractor + model) with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("SensorPipeline saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SensorPipeline":
        """Load a previously saved SensorPipeline from disk."""
        pipeline = joblib.load(path)
        logger.info("SensorPipeline loaded from %s", path)
        return pipeline
