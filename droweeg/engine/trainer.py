from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.metrics import classification_metrics, softmax


def run_backend(argv: Sequence[str] | None = None) -> None:
    from droweeg.engine import sourceonly_backend

    sourceonly_backend.main(None if argv is None else list(argv))


def preprocess_source(train_x: np.ndarray, val_x: np.ndarray | None, *, robust_clip: bool):
    clip_bounds = None
    if robust_clip:
        lo, hi = compute_clip_bounds(train_x)
        train_x = robust_clip_array(train_x, lo, hi)
        if val_x is not None:
            val_x = robust_clip_array(val_x, lo, hi)
        clip_bounds = (lo, hi)

    mean, std = compute_stats(train_x)
    state = {"clip_bounds": clip_bounds, "mean": mean, "std": std}
    train_x = zscore_array(train_x, mean, std)
    val_x = None if val_x is None else zscore_array(val_x, mean, std)
    return train_x, val_x, state


def preprocess_target(test_x: np.ndarray, state: dict[str, object]) -> np.ndarray:
    clip_bounds = state["clip_bounds"]
    if clip_bounds is not None:
        lo, hi = clip_bounds
        test_x = robust_clip_array(test_x, lo, hi)
    return zscore_array(test_x, state["mean"], state["std"])


def compute_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = x.shape[1]
    mean = x.mean(axis=(0, 2), keepdims=False).astype(np.float32).reshape(channels, 1)
    std = x.std(axis=(0, 2), keepdims=False).astype(np.float32).reshape(channels, 1)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def zscore_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    channels = x.shape[1]
    return ((x - mean.reshape(1, channels, 1)) / std.reshape(1, channels, 1)).astype(np.float32)


def compute_clip_bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = x.shape[1]
    lo = np.percentile(x, 0.5, axis=(0, 2)).astype(np.float32).reshape(channels, 1)
    hi = np.percentile(x, 99.5, axis=(0, 2)).astype(np.float32).reshape(channels, 1)
    return lo, hi


def robust_clip_array(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    channels = x.shape[1]
    return np.clip(x, lo.reshape(1, channels, 1), hi.reshape(1, channels, 1)).astype(np.float32)


def tensorize_clip_bounds(clip_bounds):
    if clip_bounds is None:
        return None
    lo, hi = clip_bounds
    return {
        "low": torch.from_numpy(lo.copy()),
        "high": torch.from_numpy(hi.copy()),
    }


def make_loader(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    x = torch.from_numpy(arrays["x"]).float().unsqueeze(1)
    y = torch.from_numpy(arrays["y"]).long()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=make_seed_worker(seed) if num_workers > 0 else None,
    )


def make_unlabeled_loader(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    x = torch.from_numpy(arrays["x"]).float().unsqueeze(1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=make_seed_worker(seed) if num_workers > 0 else None,
    )


def make_seed_worker(seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.momentum > 0,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def make_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def compute_class_weights(y: np.ndarray, class_balance: str) -> np.ndarray | None:
    if class_balance == "none":
        return None
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute weighted loss with empty class count: {counts.tolist()}")
    total = int(counts.sum())
    return (total / (2.0 * counts)).astype(np.float32)


class FocalLoss(nn.Module):
    def __init__(self, *, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_criterion(args, criterion_weight: torch.Tensor | None) -> nn.Module:
    if args.loss_type == "ce":
        return nn.CrossEntropyLoss()
    if args.loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=criterion_weight)
    if args.loss_type == "focal":
        return FocalLoss(weight=criterion_weight)
    raise ValueError(f"Unsupported loss type: {args.loss_type}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    *,
    grad_clip_norm: float,
    show_progress: bool = False,
) -> tuple[float, dict[str, object]]:
    model.train()
    total_loss = 0.0
    total_count = 0
    logits_list = []
    y_list = []
    for x, y in tqdm(loader, desc="train", leave=False, disable=not show_progress):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if hasattr(model, "apply_max_norm_constraints"):
            model.apply_max_norm_constraints()
        total_loss += float(loss.item()) * len(y)
        total_count += len(y)
        logits_list.append(logits.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(logits_list, axis=0)
    y_np = np.concatenate(y_list, axis=0)
    probs = softmax(logits_np)
    y_pred = probs.argmax(axis=1)
    metrics = classification_metrics(y_np, y_pred, probs[:, 1])
    return total_loss / max(total_count, 1), metrics


def first_shuffled_sample_ids(arrays: dict[str, np.ndarray], seed: int, n: int) -> list[str]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(arrays["y"]), generator=generator).numpy()
    return [str(arrays["sample_id"][idx]) for idx in permutation[:n]]


def model_parameter_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for _, parameter in model.state_dict().items():
            tensor = parameter.detach().cpu().contiguous()
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fit_source(*args, **kwargs):
    return train_one_epoch(*args, **kwargs)
