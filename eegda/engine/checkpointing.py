from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)


def select_checkpoint(policy: str, *, last_state=None, best_state=None, fixed_state=None):
    if policy == "last":
        return last_state
    if policy == "best_val":
        return best_state
    if policy == "fixed_epoch":
        return fixed_state
    raise ValueError(f"Unsupported checkpoint policy: {policy}")


def copy_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def monitor_value(metrics: dict[str, object], metric_name: str) -> float:
    value = float(metrics[metric_name])
    if np.isnan(value):
        return -np.inf
    return value


class CheckpointTracker:
    def __init__(
        self,
        *,
        policy: str,
        monitor_metric: str,
        min_delta: float,
        fixed_eval_epoch: int | None,
    ) -> None:
        self.policy = policy
        self.monitor_metric = monitor_metric
        self.min_delta = min_delta
        self.fixed_eval_epoch = fixed_eval_epoch

        self.best_state = None
        self.best_epoch = 0
        self.best_monitor = -1.0
        self.best_tie = -1.0
        self.best_macro_f1 = -1.0
        self.best_balanced_acc = -1.0

        self.selected_state = None
        self.selected_epoch = 0
        self.selected_reason = ""
        self.epochs_without_improvement = 0

    def update_validation(self, *, epoch: int, model, val_metrics: dict[str, object] | None) -> None:
        if val_metrics is None:
            self.epochs_without_improvement = 0
            return

        value = monitor_value(val_metrics, self.monitor_metric)
        tie_metric = "balanced_accuracy" if self.monitor_metric == "macro_f1" else "macro_f1"
        tie_value = monitor_value(val_metrics, tie_metric)
        improved = self.best_state is None or value > self.best_monitor + self.min_delta or (
            np.isclose(value, self.best_monitor) and tie_value > self.best_tie + self.min_delta
        )
        if improved:
            self.best_epoch = epoch
            self.best_monitor = value
            self.best_tie = tie_value
            self.best_macro_f1 = float(val_metrics["macro_f1"])
            self.best_balanced_acc = float(val_metrics["balanced_accuracy"])
            self.best_state = copy_model_state(model)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def update_selected(self, *, epoch: int, model) -> None:
        if self.policy == "last":
            self.selected_state = copy_model_state(model)
            self.selected_epoch = epoch
            self.selected_reason = "last"
        elif self.policy == "fixed_epoch" and epoch == self.fixed_eval_epoch:
            self.selected_state = copy_model_state(model)
            self.selected_epoch = epoch
            self.selected_reason = f"fixed_epoch_{epoch}"

    def finalize(self) -> dict[str, object]:
        if self.policy == "best_val":
            if self.best_state is None:
                raise RuntimeError("Training did not produce a best validation model")
            self.selected_state = self.best_state
            self.selected_epoch = self.best_epoch
            self.selected_reason = f"best_val_{self.monitor_metric}"
        if self.selected_state is None:
            raise RuntimeError(f"Training did not produce a checkpoint for policy={self.policy}")
        return {
            "selected_state": self.selected_state,
            "selected_epoch": self.selected_epoch,
            "selected_reason": self.selected_reason,
            "best_epoch": self.best_epoch,
            "best_monitor": self.best_monitor,
            "best_macro_f1": self.best_macro_f1,
            "best_balanced_acc": self.best_balanced_acc,
        }
