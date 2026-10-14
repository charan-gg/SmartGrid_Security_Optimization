"""
Scoring, with the traps from the project brief baked in.

Trap #4: report per-class recall and PR-AUC, never overall accuracy. This
dataset is ~71% Attack rows, so a model that always shouts "Attack" scores 0.71
accuracy and is worthless. Nothing in this module returns bare accuracy as a
headline number.
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
        # macro averages weight every class equally, so a rare class being
        # missed actually shows up instead of drowning in the Attack majority.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
    }

    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    for cls, val in zip(classes, per_class_recall):
        out[f"recall__{cls}"] = float(val)

    if y_proba is not None:
        # PR-AUC (average precision), one-vs-rest. Preferred over ROC-AUC on
        # imbalanced data because it ignores the huge true-negative mass.
        for i, cls in enumerate(classes):
            binary_truth = (np.asarray(y_true) == cls).astype(int)
            if binary_truth.sum() == 0:
                out[f"pr_auc__{cls}"] = float("nan")
                continue
            out[f"pr_auc__{cls}"] = float(average_precision_score(binary_truth, y_proba[:, i]))
        aucs = [out[f"pr_auc__{c}"] for c in classes]
        out["macro_pr_auc"] = float(np.nanmean(aucs))

    return out


def text_report(y_true, y_pred, classes=None) -> str:
    """Full sklearn report, for the log and for Appendix A."""
    classes = list(classes if classes is not None else C.CLASS_ORDER)
    return classification_report(y_true, y_pred, labels=classes, zero_division=0, digits=4)


def confusion(y_true, y_pred, classes=None):
    classes = list(classes if classes is not None else C.CLASS_ORDER)
    return confusion_matrix(y_true, y_pred, labels=classes)


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Mean and standard deviation across CV folds. The SD matters: if E1's
    per-view gap is smaller than the fold SD, there is no finding."""
    keys = fold_results[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        out[k] = float(np.nanmean(vals))
        out[f"{k}__sd"] = float(np.nanstd(vals))
    return out


def false_alerts_per_day(false_positive_rate: float) -> float:
    """Turn a model-level FPR into the number an operator actually cares about.

    A substation control room will silence a detector that cries wolf. This is
    the conversion that makes E6 an architecture result rather than a plot:
        alerts/day = FPR x (benign samples per day)
    """
    benign_samples_per_day = C.SAMPLE_RATE_HZ * 86400 * C.BENIGN_FRACTION_REAL
    return float(false_positive_rate * benign_samples_per_day)


def upstream_bytes_per_second(n_features: int, n_bays: int = 1) -> float:
    """E3's architecture output: what a feature budget costs on the wire."""
    return float(n_features * C.BYTES_PER_FEATURE * C.SAMPLE_RATE_HZ * n_bays)
