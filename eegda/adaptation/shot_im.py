from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import torch
from torch import nn


def adapt_shot_im(
    model: nn.Module,
    target_loader,
    device: torch.device,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    entropy_weight: float = 1.0,
    diversity_weight: float = 1.0,
    freeze_classifier: bool = True,
    grad_clip_norm: float = 0.0,
    log_interval: int = 10,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Run SHOT-style information maximization on unlabeled target data.

    This implements the IM-only part of SHOT for source-free adaptation:
    minimize mean prediction entropy while maximizing marginal prediction
    entropy. It does not use pseudo labels, source data, or target labels.
    By default, the classifier head is frozen and only feature parameters are
    updated.
    """

    if epochs <= 0:
        raise ValueError("shot_epochs must be positive")
    if lr <= 0:
        raise ValueError("shot_lr must be positive")
    if weight_decay < 0:
        raise ValueError("shot_weight_decay must be non-negative")
    if grad_clip_norm < 0:
        raise ValueError("shot_grad_clip_norm must be non-negative")
    if entropy_weight < 0 or diversity_weight < 0:
        raise ValueError("SHOT-IM loss weights must be non-negative")

    started_at = perf_counter()
    model.to(device)
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    before_params = _parameter_snapshot(model)
    before_bn = _bn_stats_snapshot(model)

    frozen_names = _freeze_classifier(model) if freeze_classifier else []
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("SHOT-IM found no trainable parameters after applying freeze policy")

    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    target_samples = 0
    target_batches = 0

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_entropy = 0.0
        total_diversity = 0.0
        total_count = 0
        epoch_batches = 0
        for batch in target_loader:
            x = _batch_inputs(batch).to(device)
            batch_size = int(x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
            sample_entropy = -(probs * probs.log()).sum(dim=1).mean()
            marginal = probs.mean(dim=0).clamp_min(1e-8)
            diversity_entropy = -(marginal * marginal.log()).sum()
            loss = entropy_weight * sample_entropy - diversity_weight * diversity_entropy
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_entropy += float(sample_entropy.detach().cpu().item()) * batch_size
            total_diversity += float(diversity_entropy.detach().cpu().item()) * batch_size
            total_count += batch_size
            epoch_batches += 1
            target_samples += batch_size
            target_batches += 1

        row = {
            "epoch": epoch,
            "loss": total_loss / max(total_count, 1),
            "mean_entropy": total_entropy / max(total_count, 1),
            "diversity_entropy": total_diversity / max(total_count, 1),
            "target_samples": total_count,
            "target_batches": epoch_batches,
        }
        history.append(row)
        if log_fn is not None and log_interval > 0 and (epoch == 1 or epoch % log_interval == 0 or epoch == epochs):
            log_fn(
                "SHOT-IM "
                f"epoch={epoch:03d} loss={row['loss']:.4f} "
                f"entropy={row['mean_entropy']:.4f} diversity={row['diversity_entropy']:.4f}"
            )

    model.eval()
    after_params = _parameter_snapshot(model)
    after_bn = _bn_stats_snapshot(model)
    changed_params = _changed_keys(before_params, after_params)
    frozen_changed = [name for name in frozen_names if name in changed_params]
    if frozen_changed:
        raise RuntimeError(f"SHOT-IM changed frozen classifier parameters unexpectedly: {frozen_changed}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])

    trainable_changed = [name for name in changed_params if name not in frozen_names]
    report = {
        "method": "shot_im",
        "target_samples_used": target_samples,
        "target_batches_used": target_batches,
        "target_adaptation_mode": "target_test_unlabeled",
        "labels_used_for_adaptation": False,
        "labels_ignored": True,
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "entropy_weight": float(entropy_weight),
        "diversity_weight": float(diversity_weight),
        "freeze_classifier": bool(freeze_classifier),
        "frozen_parameter_names": frozen_names,
        "frozen_parameters_changed": bool(frozen_changed),
        "trainable_parameters_changed": bool(trainable_changed),
        "trainable_parameters_changed_count": len(trainable_changed),
        "trainable_parameters_changed_names": trainable_changed,
        "bn_running_stats_changed": bool(_changed_bn_stats(before_bn, after_bn)),
        "bn_layers_updated": _changed_bn_stats(before_bn, after_bn),
        "optimizer": "adam",
        "grad_clip_norm": float(grad_clip_norm),
        "loss_history": history,
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
    raise TypeError(f"Unsupported target loader batch type for SHOT-IM: {type(batch)!r}")


def _freeze_classifier(model: nn.Module) -> list[str]:
    frozen = []
    classifier = getattr(model, "classifier", None)
    classifier_param_ids = set()
    if isinstance(classifier, nn.Module):
        classifier_param_ids = {id(parameter) for parameter in classifier.parameters()}
    for name, parameter in model.named_parameters():
        if name.startswith("classifier.") or id(parameter) in classifier_param_ids:
            parameter.requires_grad_(False)
            frozen.append(name)
        else:
            parameter.requires_grad_(True)
    return frozen


def _parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _bn_stats_snapshot(model: nn.Module) -> dict[str, dict[str, torch.Tensor | None]]:
    snapshot = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        snapshot[name] = {
            "running_mean": None if module.running_mean is None else module.running_mean.detach().cpu().clone(),
            "running_var": None if module.running_var is None else module.running_var.detach().cpu().clone(),
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
