from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an EEGNet source-only checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    fields = [
        "run_id",
        "created_at",
        "label_mode",
        "target_subject",
        "seed",
        "class_balance",
        "best_epoch",
        "best_val_metric",
        "train_subject_ids",
        "val_subject_ids",
        "model_config",
    ]
    print(f"checkpoint={checkpoint_path}")
    for field in fields:
        print(f"{field}={checkpoint.get(field)}")


if __name__ == "__main__":
    main()
