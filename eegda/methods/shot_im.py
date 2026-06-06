from __future__ import annotations

from torch import nn

from eegda.adaptation.shot_im import adapt_shot_im

from .base import BaseMethod


class ShotIMMethod(BaseMethod):
    """SHOT information maximization source-free adaptation."""

    name = "shot_im"

    def __init__(
        self,
        *,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        entropy_weight: float = 1.0,
        diversity_weight: float = 1.0,
        freeze_classifier: bool = True,
        grad_clip_norm: float = 0.0,
        log_interval: int = 10,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.freeze_classifier = freeze_classifier
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self._diagnostics: dict[str, object] = {}

    def adapt(self, model: nn.Module, target_loader_unlabeled, *, ctx) -> nn.Module:
        model, report = adapt_shot_im(
            model,
            target_loader_unlabeled,
            ctx["device"],
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            entropy_weight=self.entropy_weight,
            diversity_weight=self.diversity_weight,
            freeze_classifier=self.freeze_classifier,
            grad_clip_norm=self.grad_clip_norm,
            log_interval=self.log_interval,
            log_fn=ctx.get("log_fn"),
        )
        self._diagnostics = report
        return model

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)


ShotIM = ShotIMMethod
