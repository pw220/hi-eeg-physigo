from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from droweeg.config import kwargs_to_argv, load_config
from droweeg.engine import run_backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="DrowEEG training CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", choices=("seedvig", "sadt-balanced"), default="seedvig")
    parser.add_argument("--model", choices=("eegnet",), default="eegnet")
    parser.add_argument("--method", choices=("source_only",), default="source_only")
    parser.add_argument("--protocol", choices=("loso",), default="loso")
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument("--run-all-loso", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--sadt-balanced-path", default="data/sad-data.mat")
    parser.add_argument("--label-mode", choices=("threshold35", "strict035070"), default="threshold35")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--class-balance", choices=("none", "weighted_loss"), default="weighted_loss")
    parser.add_argument("--loss-type", choices=("ce", "weighted_ce", "focal"), default="weighted_ce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-mode", choices=("subject_split", "sample_stratified", "none"), default="subject_split")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-subject-ratio", type=float, default=0.2)
    parser.add_argument("--checkpoint-policy", choices=("best_val", "last", "fixed_epoch"), default="best_val")
    parser.add_argument("--fixed-eval-epoch", type=int, default=None)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--monitor-metric",
        choices=("macro_f1", "balanced_accuracy", "accuracy", "fatigue_f1", "roc_auc", "auprc"),
        default="macro_f1",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test-every-epochs", type=int, default=0)
    parser.add_argument("--eegnet-f1", type=int, default=8)
    parser.add_argument("--eegnet-d", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=0)
    parser.add_argument("--eegnet-temporal-kernel", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel", type=int, default=16)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-norm-rate", type=float, default=0.25)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    backend_argv = to_backend_argv(args)
    run_backend(backend_argv)
    return {"status": "completed", "backend_args": backend_argv}


def run_from_kwargs(**kwargs) -> dict[str, Any]:
    return main(kwargs_to_argv(kwargs))


def to_backend_argv(args: argparse.Namespace) -> list[str]:
    label_protocol = args.label_mode if args.dataset == "seedvig" else "rt_binary"
    output_dir = _resolve_output_dir(args.output_dir, args.dataset, args.model, args.method, label_protocol)
    backend_dataset = "seedvig" if args.dataset == "seedvig" else "sadt"
    argv = [
        "--dataset",
        backend_dataset,
        "--model",
        args.model,
        "--target-subject",
        str(args.target_subject),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--optimizer",
        args.optimizer,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--class-balance",
        args.class_balance,
        "--loss-type",
        args.loss_type,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--validation-mode",
        args.validation_mode,
        "--val-ratio",
        str(args.val_ratio),
        "--val-subject-ratio",
        str(args.val_subject_ratio),
        "--checkpoint-policy",
        args.checkpoint_policy,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--monitor-metric",
        args.monitor_metric,
        "--output-dir",
        output_dir,
        "--output-layout",
        "droweeg",
        "--test-every-epochs",
        str(args.test_every_epochs),
        "--eegnet-f1",
        str(args.eegnet_f1),
        "--eegnet-d",
        str(args.eegnet_d),
        "--eegnet-f2",
        str(args.eegnet_f2),
        "--eegnet-temporal-kernel",
        str(args.eegnet_temporal_kernel),
        "--eegnet-separable-kernel",
        str(args.eegnet_separable_kernel),
        "--eegnet-dropout",
        str(args.eegnet_dropout),
        "--eegnet-norm-rate",
        str(args.eegnet_norm_rate),
    ]
    if args.dataset == "seedvig":
        argv.extend(["--label-mode", args.label_mode])
        if args.raw_data_dir is not None:
            argv.extend(["--raw-data-dir", str(args.raw_data_dir)])
        if args.label_dir is not None:
            argv.extend(["--label-dir", str(args.label_dir)])
    else:
        argv.extend(["--sadt-path", str(args.sadt_balanced_path)])
        argv.extend(["--dataset-display-name", "sadt-balanced"])
    if args.run_all_loso:
        argv.append("--run-all-loso")
    if args.max_folds is not None:
        argv.extend(["--max-folds", str(args.max_folds)])
    if args.dry_run:
        argv.append("--dry-run")
    if args.deterministic:
        argv.append("--deterministic")
    if args.fixed_eval_epoch is not None:
        argv.extend(["--fixed-eval-epoch", str(args.fixed_eval_epoch)])
    if args.disable_early_stop:
        argv.append("--disable-early-stop")
    if args.skip_existing:
        argv.append("--skip-existing")
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def _resolve_output_dir(output_dir: str, dataset: str, model: str, method: str, label_protocol: str) -> str:
    if output_dir.strip().lower() in {"none", "null", "off", "false"}:
        return "none"
    return str(Path(output_dir) / f"{dataset}_{model}_{method}_{label_protocol}")


if __name__ == "__main__":
    main()
