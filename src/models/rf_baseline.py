import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class RFClassifier:
    """Random Forest wrapper that operates on flat feature vectors from FeatureExtractor."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFClassifier":
        """Fit the random forest on extracted feature vectors.

        Parameters
        ----------
        X : np.ndarray, shape (N, num_features)
        y : np.ndarray, shape (N,) — integer class labels [0, 5]
        """
        logger.info("Fitting RFClassifier on X=%s y=%s", X.shape, y.shape)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels, shape (N,)."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates, shape (N, 6)."""
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose sklearn feature importances for visualization."""
        return self.model.feature_importances_
