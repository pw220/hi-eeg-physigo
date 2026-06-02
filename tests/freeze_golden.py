from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eegda


GOLDEN_PATH = ROOT / "tests" / "golden_sourceonly.json"


GOLDEN_CONFIG = {
    "toy": {
        "n_subjects": 4,
        "samples_per_subject": 8,
        "channels": 3,
        "samples": 64,
        "random_state": 42,
    },
    "run": {
        "epochs": 5,
        "batch_size": 4,
        "seed": 42,
        "validation_mode": "none",
        "checkpoint_policy": "last",
        "class_balance": "weighted_loss",
        "loss_type": "weighted_ce",
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "eegnet_temporal_kernel": 16,
        "eegnet_separable_kernel": 8,
    },
}


METRIC_KEYS = [
    "target_subject",
    "target_subject_raw",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "fatigue_precision",
    "fatigue_recall",
    "specificity",
    "roc_auc",
    "auprc",
    "tn",
    "fp",
    "fn",
    "tp",
]


def main() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eegda_golden_") as tmp:
        tmpdir = Path(tmp)
        data_path = tmpdir / "toy_sourceonly.npz"
        output_dir = tmpdir / "outputs"
        toy = eegda.make_toy_dataset(**GOLDEN_CONFIG["toy"])
        toy.save(data_path)
        run = GOLDEN_CONFIG["run"]
        cmd = [
            sys.executable,
            str(ROOT / "train_eegnet_source.py"),
            "--dataset",
            "standard-npz",
            "--standard-npz-path",
            str(data_path),
            "--run-all-loso",
            "--epochs",
            str(run["epochs"]),
            "--batch-size",
            str(run["batch_size"]),
            "--device",
            "cpu",
            "--validation-mode",
            run["validation_mode"],
            "--checkpoint-policy",
            run["checkpoint_policy"],
            "--class-balance",
            run["class_balance"],
            "--loss-type",
            run["loss_type"],
            "--optimizer",
            run["optimizer"],
            "--lr",
            str(run["lr"]),
            "--weight-decay",
            str(run["weight_decay"]),
            "--seed",
            str(run["seed"]),
            "--eegnet-temporal-kernel",
            str(run["eegnet_temporal_kernel"]),
            "--eegnet-separable-kernel",
            str(run["eegnet_separable_kernel"]),
            "--output-dir",
            str(output_dir),
            "--output-layout",
            "eegda",
            "--log-level",
            "quiet",
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        metrics_path = output_dir / "metrics" / "fold_metrics.csv"
        df = pd.read_csv(metrics_path).sort_values("target_subject")
        rows = df[METRIC_KEYS].to_dict(orient="records")
        GOLDEN_PATH.write_text(
            json.dumps(
                {
                    "description": "Frozen source-only regression metrics generated from train_eegnet_source.py.",
                    "config": GOLDEN_CONFIG,
                    "metric_keys": METRIC_KEYS,
                    "fold_metrics": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
