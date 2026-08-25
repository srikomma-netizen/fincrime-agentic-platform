"""Unsupervised anomaly detection.

Uses scikit-learn's Isolation Forest when available. On locked-down hosts where
scikit-learn/scipy cannot load (e.g. Windows Application Control blocking
compiled extensions), it transparently falls back to a dependency-free,
numpy-only standardized-distance detector with the same interface.

Score is normalized to [0, 1] (1 = most anomalous).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import FEATURE_NAMES, featurize, to_vector

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

VERSION = "isoforest-v1" if _HAS_SKLEARN else "numpy-distance-v1"


class _NumpyDistanceDetector:
    """Standardized-distance anomaly detector (numpy-only fallback)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, scale: float) -> None:
        self.mean = mean
        self.std = std
        self.scale = scale  # distance at the chosen contamination quantile

    @classmethod
    def fit(cls, X: np.ndarray, contamination: float) -> "_NumpyDistanceDetector":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        z = (X - mean) / std
        dist = np.linalg.norm(z, axis=1)
        scale = float(np.quantile(dist, 1.0 - contamination)) or 1.0
        return cls(mean, std, scale)

    def anomaly(self, vec: np.ndarray) -> float:
        z = (vec - self.mean) / self.std
        dist = float(np.linalg.norm(z))
        # map so that dist == scale -> 0.5, larger -> toward 1
        return float(np.clip(dist / (2.0 * self.scale), 0.0, 1.0))


class AnomalyDetector:
    def __init__(self, model: Any, scaler: Any, kind: str) -> None:
        self.model = model
        self.scaler = scaler
        self.kind = kind

    @classmethod
    def train(cls, X: np.ndarray, contamination: float = 0.08) -> "AnomalyDetector":
        contamination = float(min(max(contamination, 0.01), 0.4))
        if _HAS_SKLEARN:
            scaler = StandardScaler().fit(X)
            model = IsolationForest(
                n_estimators=200, contamination=contamination, random_state=42
            ).fit(scaler.transform(X))
            return cls(model, scaler, "sklearn")
        model = _NumpyDistanceDetector.fit(X, contamination)
        return cls(model, None, "numpy")

    def score(self, claim: dict[str, Any]) -> float:
        vec = to_vector(featurize(claim))
        if self.kind == "sklearn":
            raw = self.model.decision_function(self.scaler.transform(vec))[0]
            return float(np.clip(0.5 - raw, 0.0, 1.0))
        return self.model.anomaly(vec[0])

    def save(self, path: Path) -> None:
        joblib.dump(
            {"model": self.model, "scaler": self.scaler,
             "kind": self.kind, "features": FEATURE_NAMES},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        blob = joblib.load(path)
        return cls(blob["model"], blob["scaler"], blob.get("kind", "sklearn"))
