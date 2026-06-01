from __future__ import annotations

from torch import nn

from .base import BaseMethod


class SourceOnlyMethod(BaseMethod):
    """Source-only method plug-in.

    Source-only does not adapt on target data. The method boundary is still
    exercised so future SFDA methods can plug in at the same point without
    owning source training, checkpointing, or target-label evaluation.
    """

    name = "source_only"

    def adapt(self, model: nn.Module, target_loader_unlabeled, *, ctx) -> nn.Module:
        return model

    def diagnostics(self) -> dict:
        return {
            "target_unlabeled_used_for_adaptation": False,
            "target_bn_stats_recomputed_on_target": False,
            "target_labels_used_for_adaptation": False,
        }


SourceOnly = SourceOnlyMethod
