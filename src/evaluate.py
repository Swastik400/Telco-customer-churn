import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, y_prob, label: str = "") -> dict:
    """Return a dict of key classification metrics."""
    metrics = {
        "label":     label,
        "f1":        round(f1_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }
    return metrics


def print_report(y_true, y_pred, label: str = ""):
    if label:
        print(f"\n=== {label} ===")
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))


# ---------------------------------------------------------------------------
# Cost-sensitive threshold tuning
# ---------------------------------------------------------------------------

def cost_matrix_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    retention_cost: float = 100.0,
    avg_clv_lost: float = 500.0,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Sweep probability thresholds and compute expected business cost.

    Cost model
    ----------
    - True Positive  (correctly predicted churner, offered retention):
        cost = retention_cost
    - False Positive (non-churner offered retention unnecessarily):
        cost = retention_cost
    - False Negative (churner missed — no offer, customer leaves):
        cost = avg_clv_lost
    - True Negative  (non-churner, no action):
        cost = 0

    Returns a DataFrame with one row per threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (tp + fp) * retention_cost + fn * avg_clv_lost
        rows.append({
            "threshold": round(t, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "total_cost": total_cost,
        })
    return pd.DataFrame(rows)


def best_threshold(cost_df: pd.DataFrame) -> float:
    """Return the threshold that minimises total business cost."""
    return cost_df.loc[cost_df["total_cost"].idxmin(), "threshold"]
