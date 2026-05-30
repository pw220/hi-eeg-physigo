from __future__ import annotations

from argparse import Namespace

from droweeg.registries import get_model


def build_model(
    model_name: str,
    input_channels: int,
    input_samples: int,
    num_classes: int,
    args: Namespace | None = None,
):
    model_factory = get_model(model_name)
    kwargs = {
        "channels": input_channels,
        "samples": input_samples,
        "num_classes": num_classes,
    }
    if args is not None and model_name == "eegnet":
        kwargs.update(
            {
                "F1": getattr(args, "eegnet_f1", 8),
                "D": getattr(args, "eegnet_d", 2),
                "F2": None if getattr(args, "eegnet_f2", 0) == 0 else getattr(args, "eegnet_f2"),
                "kernLength": getattr(args, "eegnet_temporal_kernel", 64),
                "separable_kernel_length": getattr(args, "eegnet_separable_kernel", 16),
                "dropoutRate": getattr(args, "eegnet_dropout", 0.5),
                "norm_rate": getattr(args, "eegnet_norm_rate", 0.25),
            }
        )
    return model_factory(**kwargs)
