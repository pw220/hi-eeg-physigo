from __future__ import annotations

import numpy as np

from utils.metrics import classification_metrics, softmax


def evaluate_target(model, target_loader_labeled, device):
    """Evaluate target data.

    This is the engine boundary that may receive target labels. Methods should
    receive only unlabeled target loaders before this point.
    """

    from train_eegnet_source import predict_logits

    logits, y_true = predict_logits(model, target_loader_labeled, device)
    probs = softmax(logits)
    y_pred = np.argmax(probs, axis=1)
    return classification_metrics(y_true, y_pred, probs[:, 1])


__all__ = ["classification_metrics", "evaluate_target"]
