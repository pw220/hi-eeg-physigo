from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as nnf


class SamePadConv2d(nn.Module):
    """Conv2d with explicit TensorFlow/Keras-style same padding for stride 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.kernel_size[0] - 1
        pad_w = self.kernel_size[1] - 1
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h or pad_w:
            x = nnf.pad(x, (left, right, top, bottom))
        return self.conv(x)


class EEGNet(nn.Module):
    """
    Faithful PyTorch port of the ARL EEGModels EEGNet-8,2 architecture.
    This is not a newly designed backbone.
    """

    def __init__(
        self,
        channels: int = 17,
        samples: int = 1600,
        num_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernLength: int = 64,
        separable_kernel_length: int = 16,
        dropoutRate: float = 0.5,
        norm_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if channels <= 0 or samples <= 0 or num_classes <= 0:
            raise ValueError("channels, samples, and num_classes must be positive")
        if F1 <= 0 or D <= 0:
            raise ValueError("F1 and D must be positive")
        if F2 is None:
            F2 = F1 * D
        if F2 <= 0:
            raise ValueError("F2 must be positive")
        if kernLength <= 0 or separable_kernel_length <= 0:
            raise ValueError("kernel lengths must be positive")
        if not 0.0 <= dropoutRate < 1.0:
            raise ValueError("dropoutRate must be in [0, 1)")
        if norm_rate <= 0:
            raise ValueError("norm_rate must be positive")

        self.channels = channels
        self.samples = samples
        self.num_classes = num_classes
        self.norm_rate = norm_rate
        self.config = {
            "channels": channels,
            "samples": samples,
            "num_classes": num_classes,
            "F1": F1,
            "D": D,
            "F2": F2,
            "kernLength": kernLength,
            "separable_kernel_length": separable_kernel_length,
            "dropoutRate": dropoutRate,
            "norm_rate": norm_rate,
            "depthwise_spatial_max_norm": 1.0,
        }

        self.temporal_conv = SamePadConv2d(
            1,
            F1,
            kernel_size=(1, kernLength),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Keras DepthwiseConv2D((Chans, 1), depth_multiplier=D).
        self.depthwise_spatial_conv = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        self.separable_depthwise_conv = SamePadConv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, separable_kernel_length),
            groups=F1 * D,
            bias=False,
        )
        self.separable_pointwise_conv = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            feature_dim = self._forward_features(dummy).flatten(1).shape[1]
        self.feature_dim = int(feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        print(
            "EEGNet-8,2 "
            f"input_shape=(batch, 1, {channels}, {samples}) "
            f"feature_dim={self.feature_dim} "
            f"trainable_params={self.count_trainable_parameters()} "
            f"F1={F1} D={D} F2={F2} kernLength={kernLength} "
            f"separable_kernel_length={separable_kernel_length} dropoutRate={dropoutRate}"
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_spatial_conv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable_depthwise_conv(x)
        x = self.separable_pointwise_conv(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_shape(x)
        x = self._forward_features(x)
        return self.classifier(x.flatten(1))

    def _check_input_shape(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "EEGNet expects input shape (batch, 1, channels, samples); "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                "EEGNet expects a singleton input channel dimension: "
                f"(batch, 1, {self.channels}, {self.samples}); got {tuple(x.shape)}"
            )
        if x.shape[2] == self.samples and x.shape[3] == self.channels:
            raise ValueError(
                "EEGNet received swapped EEG dimensions. Expected "
                f"(batch, 1, channels={self.channels}, samples={self.samples}); "
                f"got {tuple(x.shape)}."
            )
        if x.shape[2] != self.channels or x.shape[3] != self.samples:
            raise ValueError(
                "EEGNet expects input shape "
                f"(batch, 1, channels={self.channels}, samples={self.samples}); "
                f"got {tuple(x.shape)}"
            )

    def apply_max_norm_constraints(self) -> None:
        """Apply original EEGNet max-norm constraints after optimizer updates."""
        with torch.no_grad():
            _max_norm_(self.depthwise_spatial_conv.weight, max_norm=1.0)
            _max_norm_(self.classifier.weight, max_norm=self.norm_rate)

    def count_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _max_norm_(weight: torch.Tensor, *, max_norm: float, eps: float = 1e-8) -> None:
    flat = weight.view(weight.shape[0], -1)
    norms = flat.norm(p=2, dim=1, keepdim=True)
    desired = torch.clamp(norms, max=max_norm)
    flat.mul_(desired / (eps + norms))
