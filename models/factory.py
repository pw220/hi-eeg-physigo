from __future__ import annotations

from torch import nn

from eegda.registries import get_model, register_builtin_components


def build_model(
    model_name: str,
    channels: int,
    samples: int,
    num_classes: int,
    args,
) -> nn.Module:
    register_builtin_components()
    model_factory = get_model(model_name)
    kwargs = {
        "channels": channels,
        "samples": samples,
        "num_classes": num_classes,
    }
    if str(model_name).strip().lower() == "eegnet":
        kwargs.update(
            {
                "F1": args.eegnet_f1,
                "D": args.eegnet_d,
                "F2": None if args.eegnet_f2 == 0 else args.eegnet_f2,
                "kernLength": args.eegnet_temporal_kernel,
                "separable_kernel_length": args.eegnet_separable_kernel,
                "dropoutRate": args.eegnet_dropout,
                "norm_rate": args.eegnet_norm_rate,
                "log_summary": getattr(args, "log_level", "normal") == "debug",
            }
        )
    return model_factory(**kwargs)
