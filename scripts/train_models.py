"""Train the anomaly detector and supervised risk model on synthetic claims.

Reports evaluation metrics (precision, recall, F1, ROC-AUC) and persists both
models. Uses scikit-learn's split/metrics when available; otherwise uses
dependency-free numpy implementations so it runs on locked-down hosts too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.config import get_settings  # noqa: E402
from app.ml.anomaly import AnomalyDetector, VERSION as ANOMALY_V  # noqa: E402
from app.ml.features import featurize, to_vector  # noqa: E402
from app.ml.risk_model import RiskModel, VERSION as RISK_V  # noqa: E402

try:
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    _SK = True
except Exception:
    _SK = False


# --------------------------- numpy metric fallbacks ------------------------- #
def _split(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def _roc_auc(y, p):
    # rank-based AUC (Mann-Whitney U)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _report(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f"precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})"


def load_claims() -> list[dict]:
    path = get_settings().synthetic_dir / "claims.json"
    if not path.exists():
        raise SystemExit("No claims.json — run scripts/generate_synthetic_data.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    settings = get_settings()
    claims = load_claims()
    X = np.vstack([to_vector(featurize(c))[0] for c in claims])
    y = np.array([c["is_fraud"] for c in claims])

    if _SK:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = _split(X, y)

    # ---- supervised risk model ----
    risk = RiskModel.train(X_tr, y_tr)
    proba = np.array([risk.proba_vec(x) for x in X_te])
    preds = (proba >= 0.5).astype(int)
    print(f"=== Risk model ({RISK_V}) ===")
    if _SK:
        print(classification_report(y_te, preds, digits=3))
        print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    else:
        print(_report(y_te, preds))
        print(f"ROC-AUC: {_roc_auc(y_te, proba):.3f}")
    risk.save(settings.model_dir / "risk.joblib")

    # ---- unsupervised anomaly detector ----
    anomaly = AnomalyDetector.train(X_tr, contamination=float(np.mean(y_tr)))
    anomaly.save(settings.model_dir / "anomaly.joblib")
    a_scores = np.array([anomaly.score(c) for c in claims])
    print(f"\n=== Anomaly detector ({ANOMALY_V}) ===")
    print(f"mean anomaly score (fraud):  {a_scores[y == 1].mean():.3f}")
    print(f"mean anomaly score (normal): {a_scores[y == 0].mean():.3f}")
    print(f"\nsaved models -> {settings.model_dir}")


if __name__ == "__main__":
    main()
