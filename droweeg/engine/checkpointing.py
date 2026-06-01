from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)


def select_checkpoint(policy: str, *, last_state=None, best_state=None, fixed_state=None):
    if policy == "last":
        return last_state
    if policy == "best_val":
        return best_state
    if policy == "fixed_epoch":
        return fixed_state
    raise ValueError(f"Unsupported checkpoint policy: {policy}")
