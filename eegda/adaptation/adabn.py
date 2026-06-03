from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import torch
from torch import nn


BatchNormTypes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def adapt_adabn(
    model: nn.Module,
    target_loader,
    device: torch.device,
    *,
    reset_stats: bool = True,
    momentum: float | None = None,
    num_passes: int = 1,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Recompute BatchNorm running statistics on unlabeled target data.

    AdaBN updates only BatchNorm buffers (`running_mean`, `running_var`, and
    `num_batches_tracked`). It does not use target labels, gradients, losses, or
    optimizers, and it keeps dropout inactive by setting the full model to eval
    mode before enabling training mode only for BatchNorm modules.
    """

    if num_passes <= 0:
        raise ValueError("adabn_num_passes must be positive")

    started_at = perf_counter()
    model.to(device)
    model.eval()
    before_params = _trainable_parameter_snapshot(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    bn_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, BatchNormTypes)]
    before_bn = _bn_stats_snapshot(bn_layers)
    original_momentum = {name: module.momentum for name, module in bn_layers}
    target_samples = 0
    target_batches = 0

    if not bn_layers:
        report = {
            "method": "adabn",
            "num_bn_layers": 0,
            "num_bn_layers_updated": 0,
            "target_samples_used": 0,
            "target_batches_used": 0,
            "reset_stats": bool(reset_stats),
            "momentum": momentum,
            "num_passes": int(num_passes),
            "labels_used_for_adaptation": False,
            "trainable_parameters_changed": False,
            "bn_running_stats_changed": False,
            "warning": "AdaBN requested but no BatchNorm layers were found; model is evaluated as source-only.",
            "adaptation_time_sec": perf_counter() - started_at,
        }
        if log_fn is not None:
            log_fn(report["warning"])
        return model, report

    for _, module in bn_layers:
        if reset_stats:
            module.reset_running_stats()
        module.momentum = momentum
        module.train()

    with torch.no_grad():
        for _ in range(num_passes):
            for batch in target_loader:
                x = _batch_inputs(batch).to(device)
                target_samples += int(x.shape[0])
                target_batches += 1
                model(x)

    for name, module in bn_layers:
        module.momentum = original_momentum[name]
    model.eval()

    after_params = _named_parameter_snapshot(model, names=before_params.keys())
    after_bn = _bn_stats_snapshot(bn_layers)
    changed_params = _changed_keys(before_params, after_params)
    changed_bn = _changed_bn_stats(before_bn, after_bn)
    if changed_params:
        raise RuntimeError(f"AdaBN changed trainable parameters unexpectedly: {changed_params}")

    report = {
        "method": "adabn",
        "num_bn_layers": len(bn_layers),
        "num_bn_layers_updated": len(changed_bn),
        "bn_layer_names": [name for name, _ in bn_layers],
        "bn_layers_updated": changed_bn,
        "target_samples_used": target_samples,
        "target_batches_used": target_batches,
        "reset_stats": bool(reset_stats),
        "momentum": momentum,
        "num_passes": int(num_passes),
        "labels_used_for_adaptation": False,
        "trainable_parameters_changed": False,
        "trainable_parameters_changed_keys": [],
        "bn_running_stats_changed": bool(changed_bn),
        "target_adaptation_mode": "target_test_unlabeled",
        "labels_ignored": True,
        "adaptation_time_sec": perf_counter() - started_at,
    }
    return model, report


def _batch_inputs(batch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target loader batch type for AdaBN: {type(batch)!r}")


def _trainable_parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _named_parameter_snapshot(model: nn.Module, names) -> dict[str, torch.Tensor]:
    requested = set(names)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in requested
    }


def _bn_stats_snapshot(bn_layers: list[tuple[str, nn.Module]]) -> dict[str, dict[str, torch.Tensor | None]]:
    snapshot = {}
    for name, module in bn_layers:
        snapshot[name] = {
            "running_mean": None if module.running_mean is None else module.running_mean.detach().cpu().clone(),
            "running_var": None if module.running_var is None else module.running_var.detach().cpu().clone(),
            "num_batches_tracked": (
                None if module.num_batches_tracked is None else module.num_batches_tracked.detach().cpu().clone()
            ),
        }
    return snapshot


def _changed_keys(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    keys = sorted(set(before) | set(after))
    return [key for key in keys if key not in before or key not in after or not torch.equal(before[key], after[key])]


def _changed_bn_stats(
    before: dict[str, dict[str, torch.Tensor | None]],
    after: dict[str, dict[str, torch.Tensor | None]],
) -> list[str]:
    changed = []
    for layer_name in sorted(before):
        layer_changed = False
        for stat_name in ("running_mean", "running_var"):
            lhs = before[layer_name][stat_name]
            rhs = after[layer_name][stat_name]
            if lhs is None and rhs is None:
                continue
            if lhs is None or rhs is None or not torch.equal(lhs, rhs):
                layer_changed = True
                break
        if layer_changed:
            changed.append(layer_name)
    return changed
