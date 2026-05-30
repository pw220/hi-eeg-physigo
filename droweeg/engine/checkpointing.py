from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)
