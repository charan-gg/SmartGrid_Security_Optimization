"""
Scoring. Nothing here returns bare accuracy as a headline number: Normal is only
~42% of this capture and the three attacks together are ~55%, so accuracy hides
which attack a model misses. Per-class recall, macro-F1 and PR-AUC are reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from . import config as C


def evaluate(y_true, y_pred, y_proba=None, classes=None) -> dict:
    """The standard result dict every experiment returns."""
    classes = list(classes if classes is not None else C.CLASS_ORDER)
    out = {
        # macro averages weight every class equally, so a rare class being missed
        # (41 Fault rows, 100 Replay rows) shows up instead of drowning in Normal.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
    }
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    for cls, val in zip(classes, per_class_recall):
        out[f"recall__{cls}"] = float(val)
    if y_proba is not None:
        for i, cls in enumerate(classes):
            truth = (np.asarray(y_true) == cls).astype(int)
            out[f"pr_auc__{cls}"] = (float("nan") if truth.sum() == 0
                                     else float(average_precision_score(truth, y_proba[:, i])))
        out["macro_pr_auc"] = float(np.nanmean([out[f"pr_auc__{c}"] for c in classes]))
    return out


def text_report(y_true, y_pred, classes=None) -> str:
    classes = list(classes if classes is not None else C.CLASS_ORDER)
    return classification_report(y_true, y_pred, labels=classes, zero_division=0, digits=4)


def confusion(y_true, y_pred, classes=None):
    classes = list(classes if classes is not None else C.CLASS_ORDER)
    return confusion_matrix(y_true, y_pred, labels=classes)


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Mean and standard deviation across CV folds. The SD matters: a gap between two
    views smaller than the fold SD is not a finding."""
    out = {}
    for k in fold_results[0].keys():
        vals = [r[k] for r in fold_results]
        out[k] = float(np.nanmean(vals))
        out[f"{k}__sd"] = float(np.nanstd(vals))
    return out


def attack_coverage(result: dict) -> float:
    """Mean recall over the three attack classes: how much of the attack traffic a view
    catches. This is the per-view detection figure the grid design consumes."""
    return float(np.mean([result[f"recall__{c}"] for c in C.ATTACK_CLASSES]))


def false_alerts_per_day(false_positive_rate: float, negative_rate_hz: float) -> float:
    """Turn a model-level FPR into the number an operator cares about:
    alerts/day = FPR x (non-attack messages per day), with the rate measured from
    the capture (data.rates()['negative_rate_hz'])."""
    return float(false_positive_rate * negative_rate_hz * 86400.0)


def upstream_bytes_per_second(n_features: int, rate_hz: float, n_bays: int = 1) -> float:
    """E3's architecture output: what a feature budget costs on the wire."""
    return float(n_features * C.BYTES_PER_FEATURE * rate_hz * n_bays)
