"""Feature engineering shared by training and inference.

Keeping a single source of truth for the feature vector guarantees the model
sees identical features at train and serve time (no training/serving skew).
"""
from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_NAMES: list[str] = [
    "amount",
    "amount_to_provider_avg_ratio",
    "prior_claims_30d",
    "duplicate_flag",
    "high_dollar",
    "off_hours",  # placeholder derived signal
    "notes_len",
]


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def featurize(claim: dict[str, Any]) -> dict[str, float]:
    amount = float(claim.get("amount", 0.0))
    avg = float(claim.get("avg_provider_amount", 0.0)) or 1.0
    ratio = _safe_ratio(amount, avg)
    return {
        "amount": amount,
        "amount_to_provider_avg_ratio": ratio,
        "prior_claims_30d": float(claim.get("prior_claims_30d", 0)),
        "duplicate_flag": 1.0 if claim.get("duplicate_flag") else 0.0,
        "high_dollar": 1.0 if amount >= 10_000 else 0.0,
        "off_hours": 1.0 if str(claim.get("place_of_service", "")).lower() == "home" else 0.0,
        "notes_len": float(len(str(claim.get("notes", "")))),
    }


def to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([[feats[name] for name in FEATURE_NAMES]], dtype=float)
