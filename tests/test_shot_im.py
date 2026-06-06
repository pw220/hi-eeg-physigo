from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eegda.adaptation.shot_im import adapt_shot_im


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.classifier = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def test_shot_im_uses_unlabeled_target_and_freezes_classifier() -> None:
    torch.manual_seed(0)
    model = TinyNet()
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    _, report = adapt_shot_im(
        model,
        loader,
        torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        log_interval=1,
    )

    after = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    assert report["labels_used_for_adaptation"] is False
    assert report["freeze_classifier"] is True
    assert report["frozen_parameters_changed"] is False
    assert len(report["loss_history"]) == 2
    assert torch.equal(before["classifier.weight"], after["classifier.weight"])
    assert torch.equal(before["classifier.bias"], after["classifier.bias"])
    assert any(
        not torch.equal(before[name], after[name])
        for name in before
        if not name.startswith("classifier.")
    )
