from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from droweeg.config import kwargs_to_argv, load_config
from droweeg.datasets.base import EEGDataset
from droweeg.datasets.standard_npz import StandardDataset
from droweeg.engine import run_backend
from droweeg.results import DrowEEGResults


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="DrowEEG training CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a DrowEEG-standard .npz dataset. This is the recommended user-facing dataset input.",
    )
    parser.add_argument("--dataset", choices=("seedvig", "sadt-balanced", "standard-npz"), default="seedvig")
    parser.add_argument("--model", choices=("eegnet",), default="eegnet")
    parser.add_argument("--method", choices=("source_only",), default="source_only")
    parser.add_argument("--protocol", choices=("loso",), default="loso")
    parser.add_argument("--target-subject", default=None)
    parser.add_argument("--target-subjects", default=None, help="Comma-separated fold subject indices, e.g. 1,2,3.")
    parser.add_argument("--target-id-space", "--target-subject-id-space", choices=("canonical", "raw"), default="canonical")
    parser.add_argument("--run-all-loso", action="store_true", default=None)
    parser.add_argument("--no-run-all-loso", action="store_false", dest="run_all_loso", help=argparse.SUPPRESS)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--sadt-balanced-path", default="data/processed/sadt/sad-balance.mat")
    parser.add_argument("--path", default=None, help="Dataset path alias, mainly for --dataset standard-npz.")
    parser.add_argument("--standard-npz-path", default=None)
    parser.add_argument("--label-mode", choices=("threshold35", "strict035070"), default=None)
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
    parser.add_argument("--log-level", choices=("quiet", "normal", "verbose", "debug"), default="normal")
    parser.add_argument("--validation-mode", choices=("subject_split", "sample_stratified", "none"), default="subject_split")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-subject-ratio", type=float, default=0.2)
    parser.add_argument("--checkpoint-policy", choices=("best_val", "last", "fixed_epoch"), default=None)
    parser.add_argument("--fixed-eval-epoch", type=int, default=None)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--monitor-metric",
        choices=("macro_f1", "balanced_accuracy", "accuracy", "fatigue_f1", "roc_auc", "auprc"),
        default="macro_f1",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epoch-log-interval", type=int, default=10)
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


def main(argv: list[str] | None = None) -> DrowEEGResults:
    args = parse_args(argv)
    validate_target_selection(args)
    backend_argv = to_backend_argv(args)
    run_backend(backend_argv)
    return build_result(args, backend_argv)


def run_from_kwargs(**kwargs) -> DrowEEGResults:
    if kwargs.get("data") is not None and kwargs.get("dataset") is not None:
        raise ValueError("Provide either data or dataset, not both.")
    if kwargs.get("data") is not None:
        kwargs = dict(kwargs)
        kwargs.setdefault("dataset", "standard-npz")
        kwargs.setdefault("standard_npz_path", kwargs["data"])
        return main(kwargs_to_argv(kwargs))
    dataset_obj = kwargs.get("dataset")
    if isinstance(dataset_obj, EEGDataset):
        kwargs = dict(kwargs)
        kwargs["dataset"] = "standard-npz"
        standard_dataset = dataset_obj.to_standard_dataset()
        if isinstance(standard_dataset, StandardDataset) and standard_dataset.path is not None:
            kwargs["standard_npz_path"] = str(standard_dataset.path)
            return main(kwargs_to_argv(kwargs))
        with TemporaryDirectory(prefix="droweeg_") as tmpdir:
            path = Path(tmpdir) / "dataset.npz"
            standard_dataset.save(path)
            kwargs["standard_npz_path"] = str(path)
            return main(kwargs_to_argv(kwargs))
    return main(kwargs_to_argv(kwargs))


def to_backend_argv(args: argparse.Namespace) -> list[str]:
    dataset_name = effective_dataset(args)
    if dataset_name == "seedvig":
        label_protocol = args.label_mode or "threshold35"
    elif dataset_name == "sadt-balanced":
        label_protocol = "rt_binary"
    else:
        label_protocol = label_protocol_for_args(args)
    dataset_key = dataset_key_for_args(args)
    output_dir = _resolve_output_dir(args.output_dir, dataset_key, args.model, args.method, label_protocol)
    backend_dataset = {"seedvig": "seedvig", "sadt-balanced": "sadt", "standard-npz": "standard-npz"}[dataset_name]
    argv = [
        "--dataset",
        backend_dataset,
        "--model",
        args.model,
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
        "--log-level",
        args.log_level,
        "--validation-mode",
        args.validation_mode,
        "--val-ratio",
        str(args.val_ratio),
        "--val-subject-ratio",
        str(args.val_subject_ratio),
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
        "--epoch-log-interval",
        str(args.epoch_log_interval),
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
    if dataset_name == "seedvig":
        argv.extend(["--label-mode", args.label_mode or "threshold35"])
        if args.raw_data_dir is not None:
            argv.extend(["--raw-data-dir", str(args.raw_data_dir)])
        if args.label_dir is not None:
            argv.extend(["--label-dir", str(args.label_dir)])
    elif dataset_name == "sadt-balanced":
        argv.extend(["--sadt-path", str(args.sadt_balanced_path)])
        argv.extend(["--dataset-display-name", "sadt-balanced"])
    else:
        standard_npz_path = standard_npz_path_for_args(args)
        if standard_npz_path is None:
            raise ValueError("Standard dataset training requires --data, --standard-npz-path, or --path")
        argv.extend(["--standard-npz-path", str(standard_npz_path)])
        argv.extend(["--dataset-display-name", dataset_key])
        if args.label_mode is not None:
            argv.extend(["--label-mode", args.label_mode])
    if args.checkpoint_policy is not None:
        argv.extend(["--checkpoint-policy", args.checkpoint_policy])
    run_all_loso = effective_run_all_loso(args)
    if run_all_loso:
        argv.append("--run-all-loso")
    elif args.target_subjects is not None:
        argv.extend(["--target-subjects", target_subjects_to_cli(args.target_subjects)])
    elif args.target_subject is not None:
        argv.extend(["--target-subject", str(args.target_subject)])
    else:
        argv.extend(["--target-subject", "1"])
    argv.extend(["--target-id-space", args.target_id_space])
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


def build_result(args: argparse.Namespace, backend_argv: list[str]) -> DrowEEGResults:
    label_protocol = label_protocol_for_args(args)
    dataset_name = effective_dataset(args)
    dataset_key = dataset_key_for_args(args)
    output_dir = _resolve_output_dir(args.output_dir, dataset_key, args.model, args.method, label_protocol)
    outputs_enabled = output_dir.strip().lower() not in {"none", "null", "off", "false"}
    result: dict[str, Any] = {
        "status": "completed",
        "backend_args": backend_argv,
        "dataset": dataset_key,
        "dataset_backend": dataset_name,
        "data": standard_npz_path_for_args(args),
        "model": args.model,
        "method": args.method,
        "protocol": args.protocol,
        "label_protocol": label_protocol,
        "target_subject": args.target_subject,
        "target_subjects": None if args.target_subjects is None else parse_target_subjects_value(args.target_subjects),
        "target_id_space": args.target_id_space,
        "run_all_loso": effective_run_all_loso(args),
        "dry_run": args.dry_run,
        "outputs_enabled": outputs_enabled,
        "output_dir": None if not outputs_enabled else output_dir,
    }
    if not outputs_enabled:
        return DrowEEGResults(result)

    root = Path(output_dir)
    stem = f"{dataset_key}_{args.model}_{args.method}_{label_protocol}"
    paths = {
        "predictions_dir": root / "predictions",
        "checkpoints_dir": root / "checkpoints",
        "summaries_dir": root / "summaries",
        "reports_dir": root / "reports",
        "summary_path": root / "summaries" / f"{stem}_summary.csv",
        "manifest_path": root / "checkpoints" / "checkpoints_manifest.csv",
    }
    result.update({key: str(path) for key, path in paths.items()})
    summary_path = paths["summary_path"]
    if summary_path.exists():
        try:
            import pandas as pd

            result["summary"] = pd.read_csv(summary_path)
        except Exception as exc:  # noqa: BLE001 - result metadata should not fail training
            result["summary_load_error"] = repr(exc)
    return DrowEEGResults(result)


def label_protocol_for_args(args: argparse.Namespace) -> str:
    dataset_name = effective_dataset(args)
    if dataset_name == "seedvig":
        return args.label_mode or "threshold35"
    if dataset_name == "sadt-balanced":
        return "rt_binary"
    standard_npz_path = standard_npz_path_for_args(args)
    if standard_npz_path is not None and Path(standard_npz_path).exists():
        with np.load(standard_npz_path, allow_pickle=True) as data:
            if "metadata_json" in data:
                metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                protocol_name = metadata.get("protocol_name")
                if protocol_name:
                    return str(protocol_name)
    return "standard"


def validate_target_selection(args: argparse.Namespace) -> None:
    if args.run_all_loso and (args.target_subjects is not None or args.target_subject is not None):
        raise ValueError(
            "run_all_loso=True cannot be used with target_subject or target_subjects. "
            "Omit targets for all folds, or pass target_subject/target_subjects for selected folds."
        )


def effective_run_all_loso(args: argparse.Namespace) -> bool:
    if args.run_all_loso is not None:
        return bool(args.run_all_loso)
    return args.target_subject is None and args.target_subjects is None


def target_subjects_to_cli(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def parse_target_subjects_value(value: object) -> list[int | str]:
    if isinstance(value, (list, tuple)):
        return [int(item) if str(item).lstrip("-").isdigit() else str(item) for item in value]
    return [int(part) if part.lstrip("-").isdigit() else part for part in str(value).replace(" ", "").split(",") if part]


def effective_dataset(args: argparse.Namespace) -> str:
    return "standard-npz" if getattr(args, "data", None) is not None else args.dataset


def standard_npz_path_for_args(args: argparse.Namespace) -> str | None:
    return getattr(args, "data", None) or args.standard_npz_path or args.path


def dataset_key_for_args(args: argparse.Namespace) -> str:
    dataset_name = effective_dataset(args)
    if dataset_name != "standard-npz":
        return dataset_name
    standard_npz_path = standard_npz_path_for_args(args)
    if standard_npz_path is None:
        return "standard"
    path = Path(standard_npz_path)
    dataset_name_from_metadata = None
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            if "metadata_json" in data:
                metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                dataset_name_from_metadata = metadata.get("dataset_name")
    return _safe_name(str(dataset_name_from_metadata or path.stem))


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned.strip("_") or "dataset"


def _resolve_output_dir(output_dir: str, dataset: str, model: str, method: str, label_protocol: str) -> str:
    if output_dir.strip().lower() in {"none", "null", "off", "false"}:
        return "none"
    return str(Path(output_dir) / f"{dataset}_{model}_{method}_{label_protocol}")


if __name__ == "__main__":
    main()
