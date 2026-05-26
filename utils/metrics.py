from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(probs * np.log(probs + eps)).sum(axis=1)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= truth < 2 and 0 <= pred < 2:
            cm[truth, pred] += 1
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix_binary(y_true, y_pred)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    recalls = []
    precisions = []
    f1s = []
    for cls in (0, 1):
        tp = float(cm[cls, cls])
        fn = float(cm[cls, :].sum() - cm[cls, cls])
        fp = float(cm[:, cls].sum() - cm[cls, cls])
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "confusion_matrix": cm,
    }

