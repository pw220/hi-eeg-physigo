from __future__ import annotations

from torch import nn

from eegda.adaptation.adabn import adapt_adabn

from .base import BaseMethod


class AdaBNMethod(BaseMethod):
    """Adaptive BatchNorm source-free adaptation.

    AdaBN receives only an unlabeled target loader, recomputes BatchNorm running
    statistics, and leaves all trainable parameters unchanged.
    """

    name = "adabn"

    def __init__(
        self,
        *,
        reset_stats: bool = True,
        momentum: float | None = None,
        num_passes: int = 1,
    ) -> None:
        self.reset_stats = reset_stats
        self.momentum = momentum
        self.num_passes = num_passes
        self._diagnostics: dict[str, object] = {}

    def adapt(self, model: nn.Module, target_loader_unlabeled, *, ctx) -> nn.Module:
        model, report = adapt_adabn(
            model,
            target_loader_unlabeled,
            ctx["device"],
            reset_stats=self.reset_stats,
            momentum=self.momentum,
            num_passes=self.num_passes,
            log_fn=ctx.get("log_fn"),
        )
        self._diagnostics = report
        return model

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)


AdaBN = AdaBNMethod
