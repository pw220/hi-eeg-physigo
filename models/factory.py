from __future__ import annotations

from torch import nn

from models.eegnet import EEGNet


def build_model(
    model_name: str,
    channels: int,
    samples: int,
    num_classes: int,
    args,
) -> nn.Module:
    if model_name == "eegnet":
        return EEGNet(
            channels=channels,
            samples=samples,
            num_classes=num_classes,
            F1=args.eegnet_f1,
            D=args.eegnet_d,
            F2=None if args.eegnet_f2 == 0 else args.eegnet_f2,
            kernLength=args.eegnet_temporal_kernel,
            separable_kernel_length=args.eegnet_separable_kernel,
            dropoutRate=args.eegnet_dropout,
            norm_rate=args.eegnet_norm_rate,
        )
    raise ValueError(f"Unsupported model: {model_name}. Supported models: eegnet")
