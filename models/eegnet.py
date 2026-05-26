from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact EEGNet for binary SEED-VIG raw EEG windows."""

    def __init__(
        self,
        channels: int = 17,
        samples: int = 1600,
        num_classes: int = 2,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        temporal_kernel: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.samples = samples

        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            feature_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"EEGNet expects 4D input, got shape {tuple(x.shape)}")
        x = self.features(x)
        return self.classifier(x.flatten(1))

