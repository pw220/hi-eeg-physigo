from __future__ import annotations

import numpy as np
import torch

from utils.metrics import classification_metrics, softmax


@torch.no_grad()
def predict_logits(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    y_list = []
    for x, y in loader:
        logits = model(x.to(device))
        logits_list.append(logits.detach().cpu().numpy())
        y_list.append(y.numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(y_list, axis=0)


def evaluate_target(model, target_loader_labeled, device):
    """Evaluate target data.

    This is the engine boundary that may receive target labels. Methods should
    receive only unlabeled target loaders before this point.
    """

    logits, y_true = predict_logits(model, target_loader_labeled, device)
    probs = softmax(logits)
    y_pred = np.argmax(probs, axis=1)
    metrics = classification_metrics(y_true, y_pred, probs[:, 1])
    return {
        "logits": logits,
        "y_true": y_true,
        "probs": probs,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def evaluate_target_metrics(model, target_loader_labeled, device) -> dict[str, object]:
    return evaluate_target(model, target_loader_labeled, device)["metrics"]


__all__ = ["classification_metrics", "evaluate_target", "evaluate_target_metrics", "predict_logits"]
