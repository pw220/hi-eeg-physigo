from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(probs * np.log(probs + eps)).sum(axis=1)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    positive_label: int = 1,
) -> dict[str, object]:
    """Binary metrics with fatigue/drowsy as the positive class by default."""
    if positive_label != 1:
        raise ValueError("This project expects fatigue/drowsy to be positive class 1")

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    alert_precision = float(per_class_precision[0])
    alert_recall = float(per_class_recall[0])
    alert_f1 = float(per_class_f1[0])
    fatigue_precision = float(per_class_precision[1])
    fatigue_recall = float(per_class_recall[1])
    fatigue_f1 = float(per_class_f1[1])

    roc_auc = np.nan
    auprc = np.nan
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        if y_prob.shape != y_true.shape:
            raise ValueError(f"y_prob must match y_true shape, got {y_prob.shape} and {y_true.shape}")
        if len(np.unique(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))

    majority_class = _majority_class(y_true)
    majority_accuracy = float(np.mean(y_true == majority_class)) if len(y_true) else np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "fatigue_precision": fatigue_precision,
        "fatigue_recall": fatigue_recall,
        "fatigue_f1": fatigue_f1,
        "sensitivity": fatigue_recall,
        "alert_precision": alert_precision,
        "alert_recall": alert_recall,
        "alert_f1": alert_f1,
        "specificity": alert_recall,
        "miss_rate": 1.0 - fatigue_recall,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "majority_class": majority_class,
        "majority_accuracy": majority_accuracy,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, object]:
    return compute_binary_metrics(y_true, y_pred, y_prob, positive_label=1)


def _majority_class(y_true: np.ndarray) -> int:
    if len(y_true) == 0:
        return -1
    counts = np.bincount(y_true.astype(np.int64), minlength=2)
    return int(np.argmax(counts))
