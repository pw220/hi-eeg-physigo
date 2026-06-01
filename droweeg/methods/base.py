from __future__ import annotations

from torch import nn


class BaseMethod:
    """Method plug-in boundary.

    Engine-owned steps such as source fitting, checkpoint selection, and target
    evaluation intentionally do not live here. A method receives only unlabeled
    target data for adaptation, making target-label leakage structurally harder.
    """

    name = "base"

    def adapt(self, model: nn.Module, target_loader_unlabeled, *, ctx) -> nn.Module:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


Method = BaseMethod
