"""Supervised risk-scoring model.

Backend selection (transparent, same interface in all cases):
  1. XGBoost                      -> if installed
  2. sklearn GradientBoosting     -> if sklearn/scipy load
  3. numpy logistic regression    -> dependency-free fallback for locked-down
                                     hosts (Windows App Control blocking scipy)

Exposes per-claim feature attributions for explainability / model-risk review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import FEATURE_NAMES, featurize, to_vector

_BACKEND = "numpy-logreg"
try:
    from xgboost import XGBClassifier  # noqa: F401
    _BACKEND = "xgboost"
except Exception:  # pragma: no cover
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401
        _BACKEND = "sklearn-gbm"
    except Exception:
        _BACKEND = "numpy-logreg"

VERSION = f"risk-{_BACKEND}-v1"


class _NumpyLogReg:
    """L2-regularized logistic regression trained with gradient descent."""

    def __init__(self, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> None:
        self.w, self.b, self.mean, self.std = w, b, mean, std

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.1,
            l2: float = 1e-3) -> "_NumpyLogReg":
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
        Xs = (X - mean) / std
        n, d = Xs.shape
        w, b = np.zeros(d), 0.0
        # class weighting to handle imbalance
        pos = max(float(y.sum()), 1.0)
        neg = max(float((1 - y).sum()), 1.0)
        wpos, wneg = n / (2 * pos), n / (2 * neg)
        sw = np.where(y == 1, wpos, wneg)
        for _ in range(epochs):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = (p - y) * sw
            grad_w = Xs.T @ g / n + l2 * w
            grad_b = float(g.mean())
            w -= lr * grad_w
            b -= lr * grad_b
        return cls(w, b, mean, std)

    def proba(self, vec: np.ndarray) -> float:
        xs = (vec - self.mean) / self.std
        z = float(xs @ self.w + self.b)
        return float(1.0 / (1.0 + np.exp(-z)))

    @property
    def feature_importances_(self) -> np.ndarray:
        a = np.abs(self.w)
        return a / (a.sum() or 1.0)


class RiskModel:
    def __init__(self, model: Any, backend: str) -> None:
        self.model = model
        self.backend = backend

    @classmethod
    def train(cls, X: np.ndarray, y: np.ndarray) -> "RiskModel":
        if _BACKEND == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.1,
                              subsample=0.9, eval_metric="logloss", random_state=42)
            m.fit(X, y)
            return cls(m, _BACKEND)
        if _BACKEND == "sklearn-gbm":
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(random_state=42)
            m.fit(X, y)
            return cls(m, _BACKEND)
        return cls(_NumpyLogReg.fit(X, y.astype(float)), _BACKEND)

    def predict_proba(self, claim: dict[str, Any]) -> float:
        vec = to_vector(featurize(claim))
        if self.backend == "numpy-logreg":
            return self.model.proba(vec[0])
        return float(self.model.predict_proba(vec)[0, 1])

    def proba_vec(self, vec: np.ndarray) -> float:
        if self.backend == "numpy-logreg":
            return self.model.proba(vec)
        return float(self.model.predict_proba(vec.reshape(1, -1))[0, 1])

    def top_factors(self, claim: dict[str, Any], k: int = 3) -> list[str]:
        feats = featurize(claim)
        try:
            importances = np.asarray(self.model.feature_importances_, dtype=float)
        except Exception:
            importances = np.ones(len(FEATURE_NAMES))
        vec = to_vector(feats)[0]
        norm = np.abs(vec) / (np.abs(vec).max() or 1.0)
        contrib = importances * norm
        order = np.argsort(contrib)[::-1][:k]
        return [
            f"{FEATURE_NAMES[i]}={feats[FEATURE_NAMES[i]]:.2f} (contrib {contrib[i]:.3f})"
            for i in order if contrib[i] > 0
        ]

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "backend": self.backend,
                     "features": FEATURE_NAMES, "version": VERSION}, path)

    @classmethod
    def load(cls, path: Path) -> "RiskModel":
        blob = joblib.load(path)
        return cls(blob["model"], blob.get("backend", "numpy-logreg"))
