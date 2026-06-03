from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.seedvig_integrity import (
    IntegrityReport,
    build_seedvig_integrity_report,
    print_integrity_report_summary,
    save_integrity_csv,
    write_loso_fold_integrity_report,
)
from data.seedvig_dataset import (
    apply_channel_zscore,
    apply_robust_clip,
    compute_channel_stats,
    compute_robust_clip_bounds,
    load_seedvig_file_pairs,
    nan_inf_counts,
    parse_subject_id,
    sessions_to_arrays,
)
from data.sadt_dataset import load_sadt_arrays, sadt_counts
from eegda.datasets.standard_npz import load_standard_dataset, standard_counts
from eegda.engine import checkpointing as engine_checkpointing
from eegda.engine import evaluator as engine_evaluator
from eegda.engine import trainer as engine_trainer
from eegda.protocols import loso as loso_protocol
from eegda.registries import get_method, register_builtin_components
from models.factory import build_model
from utils.metrics import classification_metrics, entropy_from_probs, softmax
from utils.seed import set_seed


LOG_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2, "debug": 3}
CORE_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "fatigue_precision",
    "fatigue_recall",
    "specificity",
    "roc_auc",
    "auprc",
]
DISPLAY_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "fatigue_precision",
    "fatigue_recall",
    "specificity",
    "roc_auc",
    "auprc",
]


@dataclass(frozen=True)
class FoldPlan:
    dataset: str
    model_name: str
    label_protocol: str
    input_channels: int
    input_samples: int
    num_classes: int
    target_subject: int
    target_subject_raw: object
    train_pairs: list[tuple[Path, Path]]
    val_pairs: list[tuple[Path, Path]]
    test_pairs: list[tuple[Path, Path]]
    train_subject_ids: list[int]
    val_subject_ids: list[int]
    test_subject_ids: list[int]
    train_subject_raw_ids: list[object]
    val_subject_raw_ids: list[object]
    test_subject_raw_ids: list[object]
    train_counts: dict[str, int]
    val_counts: dict[str, int]
    test_counts: dict[str, int]
    prediction_path: Path
    checkpoint_path: Path
    latest_checkpoint_path: Path | None
    summary_path: Path
    fold_report_path: Path
    val_metrics_path: Path
    test_metrics_path: Path
    manifest_path: Path
    run_id: str
    created_at: str
    command: str
    single_fold_command: str
    validation_mode: str
    checkpoint_policy: str
    validation_strategy: str
    outputs_enabled: bool


@dataclass(frozen=True)
class DatasetContext:
    dataset: str
    model_name: str
    label_protocol: str
    input_channels: int
    input_samples: int
    num_classes: int
    subjects: list[int]
    subject_mapping: dict[int, object]
    integrity_report: IntegrityReport | None
    file_pairs: list[tuple[Path, Path]]
    sadt_arrays: dict[str, np.ndarray] | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-only EEGNet LOSO baseline on EEG fatigue datasets")
    parser.add_argument("--dataset", choices=("seedvig", "sadt", "standard-npz"), default="seedvig")
    parser.add_argument("--dataset-display-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--method", default="source_only")
    parser.add_argument("--adaptation-protocol", choices=("none", "transductive", "inductive_split"), default="none")
    parser.add_argument("--reuse-source", action="store_true")
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--source-checkpoint-dir", default=None)
    parser.add_argument("--require-source-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-require-source-checkpoint", action="store_false", dest="require_source_checkpoint")
    parser.add_argument("--save-adapted-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-adapted-checkpoint", action="store_false", dest="save_adapted_checkpoint")
    parser.add_argument("--adabn-reset-stats", action="store_true", default=True)
    parser.add_argument("--no-adabn-reset-stats", action="store_false", dest="adabn_reset_stats")
    parser.add_argument("--adabn-momentum", type=float, default=None)
    parser.add_argument("--adabn-num-passes", type=int, default=1)
    parser.add_argument("--data-root", default="data/raw/SEED-VIG")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--sadt-path", default="data/processed/sadt/sad-data.mat")
    parser.add_argument("--standard-npz-path", default=None)
    parser.add_argument("--target-subject", default="1")
    parser.add_argument(
        "--target-subjects",
        default=None,
        help="Comma-separated 1-based fold subject indices, e.g. 1,2,3. Mutually exclusive with --run-all-loso.",
    )
    parser.add_argument("--target-id-space", choices=("canonical", "raw"), default="canonical")
    parser.add_argument("--run-all-loso", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--label-mode", choices=("threshold35", "strict035070"), default=None)
    parser.add_argument("--class-balance", choices=("none", "weighted_loss"), default="weighted_loss")
    parser.add_argument(
        "--loss-type",
        choices=("ce", "weighted_ce", "focal"),
        default="weighted_ce",
        help="Loss function. weighted_ce preserves the original weighted CrossEntropyLoss behavior.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--monitor-metric",
        choices=("macro_f1", "balanced_accuracy", "accuracy", "fatigue_f1", "roc_auc", "auprc"),
        default="macro_f1",
    )
    parser.add_argument("--val-subject-ratio", type=float, default=0.2)
    parser.add_argument("--validation-mode", choices=("subject_split", "sample_stratified", "none"), default="subject_split")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--checkpoint-policy", choices=("best_val", "last", "fixed_epoch"), default=None)
    parser.add_argument("--fixed-eval-epoch", type=int, default=None)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument(
        "--test-every-epochs",
        type=int,
        default=0,
        help="Diagnostic target-test evaluation interval. 0 means evaluate target only once at the end.",
    )
    parser.add_argument("--bandpass", action="store_true")
    parser.add_argument("--robust-clip", action="store_true")
    parser.add_argument("--eegnet-f1", type=int, default=8)
    parser.add_argument("--eegnet-d", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=0, help="0 means f1*d")
    parser.add_argument("--eegnet-temporal-kernel", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel", type=int, default=16)
    parser.add_argument("--eegnet-pool1", type=int, default=4)
    parser.add_argument("--eegnet-pool2", type=int, default=8)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-norm-rate", type=float, default=0.25)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--outputs-dir", dest="output_dir", help=argparse.SUPPRESS)
    parser.add_argument("--output-layout", choices=("flat", "eegda", "droweeg"), default="flat", help=argparse.SUPPRESS)
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--debug-repro", action="store_true")
    parser.add_argument("--log-level", choices=("quiet", "normal", "verbose", "debug"), default="normal")
    parser.add_argument("--epoch-log-interval", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--min-class-samples", type=int, default=1)
    args = parser.parse_args(argv)
    raw_argv = sys.argv[1:] if argv is None else argv
    args.label_mode_explicit = _argument_was_provided(raw_argv, "--label-mode")
    args.checkpoint_policy_explicit = _argument_was_provided(raw_argv, "--checkpoint-policy")
    resolve_protocol_defaults(args)
    return args


def make_method(args: argparse.Namespace):
    method_cls = get_method(args.method)
    if args.method == "adabn":
        return method_cls(
            reset_stats=args.adabn_reset_stats,
            momentum=args.adabn_momentum,
            num_passes=args.adabn_num_passes,
        )
    return method_cls()


def _argument_was_provided(argv: list[str], name: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in argv)


def resolve_protocol_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "seedvig" and args.label_mode is None:
        args.label_mode = "threshold35"
    if args.method == "adabn" and args.adaptation_protocol == "none":
        args.adaptation_protocol = "transductive"
    if args.checkpoint_policy is None:
        args.checkpoint_policy = "last" if args.validation_mode == "none" else "best_val"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    if (args.raw_data_dir is None) != (args.label_dir is None):
        raise ValueError("--raw-data-dir and --label-dir must be provided together")
    validate_training_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    outputs_dir = Path(args.output_dir)
    if outputs_enabled(args):
        outputs_dir.mkdir(parents=True, exist_ok=True)

    context = build_dataset_context(args, outputs_dir)
    target_subjects = resolve_target_subjects(args, context.subjects, context.subject_mapping)
    plans = [
        plan_loso_fold(
            args=args,
            context=context,
            target_subject=target_subject,
            outputs_dir=outputs_dir,
        )
        for target_subject in target_subjects
    ]
    device = choose_device(args.device)
    repro_metadata = reproducibility_metadata(args, device)
    write_run_reports(args, context, plans, target_subjects, repro_metadata)
    if should_log(args, "normal"):
        print_run_overview(args, context, target_subjects, plans, device)
    if should_log(args, "debug"):
        print_global_plan_header(args, context, target_subjects)
        print_model_selection_policy(args)
        print_reproducibility_metadata(repro_metadata)
    elif should_log(args, "verbose"):
        print_model_selection_policy(args)

    if args.dry_run:
        for idx, plan in enumerate(plans, start=1):
            if should_log(args, "verbose"):
                print_fold_plan(plan, dry_run=True)
            elif should_log(args, "normal"):
                print_compact_fold_start(plan, idx, len(plans), dry_run=True)
        if should_log(args, "debug"):
            print_recommended_gpu_command(args)
        return

    for idx, plan in enumerate(plans, start=1):
        if args.skip_existing and plan.outputs_enabled and fold_outputs_exist(plan):
            console(args, f"Skipping target_subject={plan.target_subject}: existing prediction CSV and checkpoint found", "normal")
            continue
        try:
            if args.reuse_source:
                run_reuse_source_fold(args, context, plan, device, repro_metadata, fold_index=idx, fold_total=len(plans))
            else:
                run_loso_fold(args, context, plan, device, repro_metadata, fold_index=idx, fold_total=len(plans))
        except Exception as exc:  # noqa: BLE001 - all-LOSO should continue after fold failures
            print(f"Fold target_subject={plan.target_subject} failed: {exc}")
            traceback.print_exc()
            if plan.outputs_enabled:
                write_failed_summary_row(plan.summary_path, args, plan, exc)
                try:
                    write_checkpoint_manifest_row(
                        plan.manifest_path,
                        args,
                        plan,
                        status="failed",
                        best_epoch=None,
                        best_val_metric=None,
                        error=repr(exc),
                    )
                except Exception as manifest_exc:  # noqa: BLE001
                    print(f"Could not write failed manifest row for target_subject={plan.target_subject}: {manifest_exc}")
            if not args.run_all_loso:
                raise

    print_final_aggregate_summary(args, plans)
    if should_log(args, "debug"):
        print_recommended_gpu_command(args)


def resolve_target_subjects(
    args: argparse.Namespace,
    subjects: list[int],
    subject_mapping: dict[int, object],
) -> list[int]:
    if args.run_all_loso and args.target_subjects:
        raise ValueError(
            "--run-all-loso cannot be used with --target-subjects. "
            "Use --run-all-loso for every fold, or --target-subjects for a selected subset."
        )
    if args.target_subjects:
        selected = resolve_subject_tokens(parse_target_subjects(args.target_subjects), args.target_id_space, subjects, subject_mapping)
        return selected
    if args.run_all_loso:
        selected = subjects
        if args.max_folds is not None:
            selected = selected[: args.max_folds]
        if not selected:
            raise ValueError("No target subjects available for --run-all-loso")
        return selected
    return [resolve_subject_token(str(args.target_subject), args.target_id_space, subjects, subject_mapping)]


def parse_target_subjects(raw: str) -> list[str]:
    values = []
    for part in str(raw).replace(" ", "").split(","):
        if not part:
            continue
        value = part
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--target-subjects was provided but no subjects were parsed")
    return values


def resolve_subject_tokens(
    tokens: list[str],
    id_space: str,
    subjects: list[int],
    subject_mapping: dict[int, object],
) -> list[int]:
    selected = [resolve_subject_token(token, id_space, subjects, subject_mapping) for token in tokens]
    deduped = []
    for subject in selected:
        if subject not in deduped:
            deduped.append(subject)
    return deduped


def resolve_subject_token(
    token: str,
    id_space: str,
    subjects: list[int],
    subject_mapping: dict[int, object],
) -> int:
    if id_space == "canonical":
        try:
            subject = int(token)
        except ValueError as exc:
            raise ValueError(f"Canonical target subject IDs must be integers. {valid_subject_message(subjects, subject_mapping)}") from exc
        if subject not in subjects:
            raise ValueError(f"Target subject {subject} not found. {valid_subject_message(subjects, subject_mapping)}")
        return subject
    for canonical_id, raw_id in subject_mapping.items():
        if raw_id_matches(token, raw_id):
            return int(canonical_id)
    raise ValueError(f"Raw target subject {token!r} not found. {valid_subject_message(subjects, subject_mapping)}")


def raw_id_matches(token: str, raw_id: object) -> bool:
    if token == str(raw_id):
        return True
    try:
        return float(token) == float(raw_id)
    except (TypeError, ValueError):
        return False


def valid_subject_message(subjects: list[int], subject_mapping: dict[int, object]) -> str:
    raw_ids = [subject_mapping.get(subject, subject) for subject in subjects]
    return f"Valid canonical IDs: {subjects}; valid raw IDs: {raw_ids}."


def should_log(args: argparse.Namespace, level: str) -> bool:
    return LOG_LEVELS[args.log_level] >= LOG_LEVELS[level]


def console(args: argparse.Namespace, message: str = "", level: str = "normal") -> None:
    if should_log(args, level):
        print(message)


def build_dataset_context(args: argparse.Namespace, outputs_dir: Path) -> DatasetContext:
    if args.dataset == "seedvig":
        reports = {}
        for label_mode in ("threshold35", "strict035070"):
            report = build_seedvig_integrity_report(
                args.data_root,
                raw_data_dir=args.raw_data_dir,
                label_dir=args.label_dir,
                label_mode=label_mode,
                min_class_samples=args.min_class_samples,
                metadata_only=args.dry_run,
            )
            reports[label_mode] = report
            if not args.dry_run and outputs_enabled(args):
                integrity_dir = outputs_dir / "reports" if args.output_layout in {"eegda", "droweeg"} else outputs_dir
                save_integrity_csv(report, integrity_dir / f"seedvig_integrity_{label_mode}.csv")
        report = reports[args.label_mode]
        file_pairs = report.valid_file_pairs
        subjects = sorted({parse_subject_id(raw_path) for raw_path, _ in file_pairs})
        return DatasetContext(
            dataset="seedvig",
            model_name=args.model,
            label_protocol=args.label_mode,
            input_channels=17,
            input_samples=1600,
            num_classes=2,
            subjects=subjects,
            subject_mapping={int(subject): int(subject) for subject in subjects},
            integrity_report=report,
            file_pairs=file_pairs,
            sadt_arrays=None,
        )
    if args.dataset == "sadt":
        arrays = load_sadt_arrays(args.sadt_path)
        subjects = sorted({int(subject) for subject in arrays["subject_id"]})
        if should_log(args, "debug"):
            print("sadt_dataset_summary")
            print(f"  path={args.sadt_path}")
            print(f"  samples={len(arrays['y'])}")
            print(f"  subjects={subjects}")
            print("  label_protocol=rt_binary")
            print("  label_mode_not_applicable=True")
        return DatasetContext(
            dataset=args.dataset_display_name or "sadt",
            model_name=args.model,
            label_protocol="rt_binary",
            input_channels=30,
            input_samples=384,
            num_classes=2,
            subjects=subjects,
            subject_mapping={int(subject): int(subject) for subject in subjects},
            integrity_report=None,
            file_pairs=[],
            sadt_arrays=arrays,
        )
    if args.dataset == "standard-npz":
        arrays, metadata = load_standard_dataset(args.standard_npz_path)
        if "y" not in arrays:
            raise ValueError("standard-npz source-only training requires y labels")
        labels = set(np.unique(arrays["y"]).astype(int).tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f"standard-npz source-only metrics currently require binary labels {{0, 1}}, got {sorted(labels)}")
        subjects = sorted({int(subject) for subject in arrays["subject_id"]})
        subject_mapping = subject_mapping_from_arrays(arrays)
        if should_log(args, "debug"):
            print("standard_npz_dataset_summary")
            print(f"  path={args.standard_npz_path}")
            print(f"  samples={len(arrays['y'])}")
            print(f"  fold_subjects={subjects}")
            print(f"  subject_mapping={subject_mapping}")
        standard_metadata = metadata.get("metadata", {})
        label_protocol = str(standard_metadata.get("protocol_name", "standard"))
        if should_log(args, "debug"):
            print(f"  label_protocol={label_protocol}")
        return DatasetContext(
            dataset=args.dataset_display_name or "standard-npz",
            model_name=args.model,
            label_protocol=label_protocol,
            input_channels=int(arrays["x"].shape[1]),
            input_samples=int(arrays["x"].shape[2]),
            num_classes=int(max(labels)) + 1,
            subjects=subjects,
            subject_mapping=subject_mapping,
            integrity_report=None,
            file_pairs=[],
            sadt_arrays=arrays,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def validate_training_args(args: argparse.Namespace) -> None:
    register_builtin_components()
    get_method(args.method)
    if args.method not in {"source_only", "adabn"}:
        raise ValueError("Only method='source_only' and method='adabn' are implemented; no other SFDA methods are available yet.")
    if args.method == "source_only" and args.adaptation_protocol != "none":
        raise ValueError("source_only requires adaptation_protocol='none'.")
    if args.method == "source_only" and args.reuse_source:
        raise ValueError("--reuse-source is only valid for adaptation methods such as --method adabn")
    if args.method == "adabn" and args.adaptation_protocol != "transductive":
        raise ValueError("AdaBN requires adaptation_protocol='transductive'.")
    if args.reuse_source and args.method != "source_only" and args.source_manifest is None and args.source_checkpoint_dir is None:
        raise ValueError("--reuse-source requires --source-manifest or --source-checkpoint-dir")
    if args.adabn_num_passes <= 0:
        raise ValueError("--adabn-num-passes must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be non-negative")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")
    if not 0.0 < args.plateau_factor < 1.0:
        raise ValueError("--plateau-factor must be between 0 and 1")
    if args.plateau_patience < 0:
        raise ValueError("--plateau-patience must be non-negative")
    if args.min_lr < 0:
        raise ValueError("--min-lr must be non-negative")
    if not 0.0 < args.val_subject_ratio < 1.0:
        raise ValueError("--val-subject-ratio must be between 0 and 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.validation_mode == "none" and args.checkpoint_policy == "best_val":
        raise ValueError("best_val checkpoint policy requires validation-mode != none.")
    if args.checkpoint_policy == "fixed_epoch":
        if args.fixed_eval_epoch is None:
            raise ValueError("--checkpoint-policy fixed_epoch requires --fixed-eval-epoch")
        if not 1 <= args.fixed_eval_epoch <= args.epochs:
            raise ValueError("--fixed-eval-epoch must be between 1 and --epochs")
    if args.validation_mode == "none" and args.early_stop_patience > 0 and not args.disable_early_stop:
        print("warning: validation-mode=none disables early stopping because no validation metrics are computed")
    if args.test_every_epochs < 0:
        raise ValueError("--test-every-epochs must be non-negative; use 0 for final-only target evaluation")
    if args.epoch_log_interval <= 0:
        raise ValueError("--epoch-log-interval must be a positive integer")
    if args.dataset in {"sadt", "standard-npz"} and args.bandpass:
        raise ValueError("--bandpass is not supported for pre-windowed array datasets")
    if args.dataset == "standard-npz" and args.standard_npz_path is None:
        raise ValueError("--dataset standard-npz requires --standard-npz-path")
    if output_dir_is_disabled(args.output_dir) and args.skip_existing:
        raise ValueError("--skip-existing requires output saving; use an output directory instead of --output-dir none")
    if args.loss_type == "weighted_ce" and args.class_balance == "none":
        raise ValueError("--loss-type weighted_ce requires --class-balance weighted_loss")
    if args.eegnet_f1 <= 0 or args.eegnet_d <= 0:
        raise ValueError("--eegnet-f1 and --eegnet-d must be positive")
    if args.eegnet_f2 < 0:
        raise ValueError("--eegnet-f2 must be non-negative; use 0 for f1*d")
    if args.eegnet_temporal_kernel <= 0 or args.eegnet_separable_kernel <= 0:
        raise ValueError("EEGNet kernel sizes must be positive")
    if args.eegnet_pool1 <= 0 or args.eegnet_pool2 <= 0:
        raise ValueError("EEGNet pool sizes must be positive")
    if args.eegnet_pool1 != 4 or args.eegnet_pool2 != 8:
        raise ValueError("Faithful EEGNet-8,2 uses fixed pool sizes: --eegnet-pool1 4 --eegnet-pool2 8")
    if not 0.0 <= args.eegnet_dropout < 1.0:
        raise ValueError("--eegnet-dropout must be in [0, 1)")
    if args.eegnet_norm_rate <= 0:
        raise ValueError("--eegnet-norm-rate must be positive")


def output_dir_is_disabled(output_dir: str | None) -> bool:
    return output_dir is not None and str(output_dir).strip().lower() in {"none", "null", "off", "false"}


def outputs_enabled(args: argparse.Namespace) -> bool:
    return not output_dir_is_disabled(args.output_dir)


def make_run_id(args: argparse.Namespace, target_subject: int) -> str:
    base = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{base}_subject{target_subject}"


def fold_outputs_exist(plan: FoldPlan) -> bool:
    if not plan.outputs_enabled:
        return False
    return plan.prediction_path.exists() and any(existing_checkpoints_for_plan(plan))


def existing_checkpoints_for_plan(plan: FoldPlan) -> list[Path]:
    if not plan.outputs_enabled:
        return []
    checkpoints = [plan.checkpoint_path] if plan.checkpoint_path.exists() else []
    if plan.latest_checkpoint_path is not None and plan.latest_checkpoint_path.exists():
        checkpoints.append(plan.latest_checkpoint_path)
    return checkpoints


def plan_seedvig_splits(args: argparse.Namespace, context: DatasetContext, target_subject: int):
    return loso_protocol.plan_seedvig_splits(args, context, target_subject)


def plan_array_splits(args: argparse.Namespace, context: DatasetContext, target_subject: int):
    return loso_protocol.plan_array_splits(args, context, target_subject)


def array_counts(arrays: dict[str, np.ndarray], context: DatasetContext) -> dict[str, int]:
    if context.label_protocol == "rt_binary":
        return sadt_counts(arrays)
    return standard_counts(arrays)


def subject_mapping_from_arrays(arrays: dict[str, np.ndarray]) -> dict[int, object]:
    raw_subjects = arrays.get("subject_id_raw", arrays["subject_id"])
    mapping: dict[int, object] = {}
    for subject_index, raw_subject in zip(arrays["subject_id"], raw_subjects, strict=True):
        mapping.setdefault(int(subject_index), python_scalar(raw_subject))
    return dict(sorted(mapping.items()))


def raw_subject_ids(context: DatasetContext, subject_ids: list[int]) -> list[object]:
    return [context.subject_mapping.get(int(subject_id), int(subject_id)) for subject_id in subject_ids]


def python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def validation_strategy_text(validation_mode: str) -> str:
    if validation_mode == "subject_split":
        return "deterministic source-subject split controlled by seed and val_subject_ratio"
    if validation_mode == "sample_stratified":
        return "sample-level stratified validation within source subjects controlled by seed and val_ratio"
    if validation_mode == "none":
        return "no validation set; all non-target source samples used for training"
    raise ValueError(f"Unsupported validation mode: {validation_mode}")


def plan_loso_fold(
    *,
    args: argparse.Namespace,
    context: DatasetContext,
    target_subject: int,
    outputs_dir: Path,
) -> FoldPlan:
    if context.dataset == "seedvig":
        train_pairs, val_pairs, test_pairs, train_counts, val_counts, test_counts, train_subject_ids, val_subject_ids = (
            plan_seedvig_splits(args, context, target_subject)
        )
    else:
        train_pairs, val_pairs, test_pairs = [], [], []
        train_counts, val_counts, test_counts, train_subject_ids, val_subject_ids = plan_array_splits(
            args,
            context,
            target_subject,
        )
    validation_strategy = validation_strategy_text(args.validation_mode)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = make_run_id(args, target_subject)
    checkpoints_dir = outputs_dir / "checkpoints"
    output_stem = f"{context.dataset}_{context.model_name}_{args.method}_{context.label_protocol}_subject_{target_subject}"
    summary_stem = f"{context.dataset}_{context.model_name}_{args.method}_{context.label_protocol}"
    if args.output_layout in {"eegda", "droweeg"}:
        prediction_path = outputs_dir / "predictions" / f"{output_stem}.csv"
        summary_path = outputs_dir / "summaries" / f"{summary_stem}_summary.csv"
        fold_report_path = outputs_dir / "reports" / f"loso_fold_integrity_{summary_stem}_subject_{target_subject}.txt"
        val_metrics_path = outputs_dir / "reports" / f"val_metrics_{summary_stem}_subject_{target_subject}.csv"
        test_metrics_path = outputs_dir / "reports" / f"test_metrics_history_{summary_stem}_subject_{target_subject}.csv"
        manifest_path = outputs_dir / "checkpoints" / "checkpoints_manifest.csv"
    else:
        prediction_path = outputs_dir / f"{output_stem}.csv"
        summary_path = outputs_dir / f"{summary_stem}_summary.csv"
        fold_report_path = outputs_dir / f"loso_fold_integrity_{summary_stem}_subject_{target_subject}.txt"
        val_metrics_path = outputs_dir / f"val_metrics_{summary_stem}_subject_{target_subject}.csv"
        test_metrics_path = outputs_dir / f"test_metrics_history_{summary_stem}_subject_{target_subject}.csv"
        manifest_path = outputs_dir / "checkpoints_manifest.csv"
    checkpoint_path = checkpoints_dir / (
        f"{output_stem}_seed{args.seed}_{run_id}.pt"
    )
    latest_checkpoint_path = (
        checkpoints_dir / f"{output_stem}_seed{args.seed}_latest.pt"
        if args.save_latest
        else None
    )
    command = (
        "python train_eegnet_source.py "
        f"--dataset {args.dataset} "
        f"--model {args.model} "
        f"--method {args.method} "
        f"--adaptation-protocol {args.adaptation_protocol} "
        f"--target-subject {target_subject} "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        f"--lr {args.lr} "
        f"--optimizer {args.optimizer} "
        f"--weight-decay {args.weight_decay} "
        f"--grad-clip-norm {args.grad_clip_norm} "
        f"--lr-scheduler {args.lr_scheduler} "
        f"--plateau-factor {args.plateau_factor} "
        f"--plateau-patience {args.plateau_patience} "
        f"--min-lr {args.min_lr} "
        f"--early-stop-patience {args.early_stop_patience} "
        f"--min-delta {args.min_delta} "
        f"--monitor-metric {args.monitor_metric} "
        f"--validation-mode {args.validation_mode} "
        f"--val-ratio {args.val_ratio} "
        f"--val-subject-ratio {args.val_subject_ratio} "
        f"--checkpoint-policy {args.checkpoint_policy} "
        f"--test-every-epochs {args.test_every_epochs} "
        f"--epoch-log-interval {args.epoch_log_interval} "
        f"--device {args.device} "
        f"--class-balance {args.class_balance} "
        f"--loss-type {args.loss_type} "
        f"--eegnet-f1 {args.eegnet_f1} "
        f"--eegnet-d {args.eegnet_d} "
        f"--eegnet-f2 {args.eegnet_f2} "
        f"--eegnet-temporal-kernel {args.eegnet_temporal_kernel} "
        f"--eegnet-separable-kernel {args.eegnet_separable_kernel} "
        f"--eegnet-pool1 {args.eegnet_pool1} "
        f"--eegnet-pool2 {args.eegnet_pool2} "
        f"--eegnet-dropout {args.eegnet_dropout} "
        f"--eegnet-norm-rate {args.eegnet_norm_rate} "
        f"--output-dir {shlex.quote(str(args.output_dir))}"
    )
    if args.label_mode is not None:
        command += f" --label-mode {args.label_mode}"
    command += f" --target-id-space canonical"
    if args.raw_data_dir is not None and args.label_dir is not None:
        command += f" --raw-data-dir {shlex.quote(str(args.raw_data_dir))}"
        command += f" --label-dir {shlex.quote(str(args.label_dir))}"
    if args.dataset == "sadt":
        command += f" --sadt-path {shlex.quote(str(args.sadt_path))}"
    if args.dataset == "standard-npz":
        command += f" --standard-npz-path {shlex.quote(str(args.standard_npz_path))}"
    if args.bandpass:
        command += " --bandpass"
    if args.robust_clip:
        command += " --robust-clip"
    if args.fixed_eval_epoch is not None:
        command += f" --fixed-eval-epoch {args.fixed_eval_epoch}"
    if args.disable_early_stop:
        command += " --disable-early-stop"
    if args.deterministic:
        command += " --deterministic"

    return FoldPlan(
        dataset=context.dataset,
        model_name=context.model_name,
        label_protocol=context.label_protocol,
        input_channels=context.input_channels,
        input_samples=context.input_samples,
        num_classes=context.num_classes,
        target_subject=target_subject,
        target_subject_raw=context.subject_mapping.get(int(target_subject), int(target_subject)),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=[target_subject],
        train_subject_raw_ids=raw_subject_ids(context, train_subject_ids),
        val_subject_raw_ids=raw_subject_ids(context, val_subject_ids),
        test_subject_raw_ids=raw_subject_ids(context, [target_subject]),
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        prediction_path=prediction_path,
        checkpoint_path=checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        summary_path=summary_path,
        fold_report_path=fold_report_path,
        val_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        manifest_path=manifest_path,
        run_id=run_id,
        created_at=created_at,
        command=" ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        single_fold_command=command,
        validation_mode=args.validation_mode,
        checkpoint_policy=args.checkpoint_policy,
        validation_strategy=validation_strategy,
        outputs_enabled=outputs_enabled(args),
    )


def run_loso_fold(
    args: argparse.Namespace,
    context: DatasetContext,
    plan: FoldPlan,
    device: torch.device,
    repro_metadata: dict[str, object],
    *,
    fold_index: int,
    fold_total: int,
) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    guard_run_outputs(args, plan)
    write_fold_report(args, context, plan)
    if should_log(args, "debug"):
        print_dataset_fold_summary(args, context, plan)
        print_fold_plan(plan, dry_run=False)
    elif should_log(args, "verbose"):
        print_fold_plan(plan, dry_run=False)
    elif should_log(args, "normal"):
        print_compact_fold_start(plan, fold_index, fold_total, dry_run=False)

    train, val, test, train_sessions, val_sessions, test_sessions = load_fold_arrays(args, context, plan)
    fold_audit = initial_fold_audit(args, plan, train, val, test)
    assert plan.target_subject not in set(train["subject_id"])
    if val is not None:
        assert plan.target_subject not in set(val["subject_id"])

    if should_log(args, "debug"):
        print_source_sanity(train_sessions, val_sessions, context)
        print_source_split_sanity(train, val)
    elif should_log(args, "verbose"):
        print_source_split_sanity(train, val)
    train_x, val_x, preprocess_state = engine_trainer.preprocess_source(
        train["x"],
        None if val is None else val["x"],
        robust_clip=args.robust_clip,
    )
    train["x"] = train_x
    if val is not None:
        val["x"] = val_x
        if should_log(args, "debug"):
            print_nan_inf_after_preprocessing(("train", train), ("val", val))
    else:
        if should_log(args, "debug"):
            print_nan_inf_after_preprocessing(("train", train))

    # Target labels are loaded after source-only preprocessing state is fixed;
    # they are used only for final evaluation and saved prediction diagnostics.
    assert set(test["subject_id"]) == {plan.target_subject}
    test["x"] = engine_trainer.preprocess_target(test["x"], preprocess_state)
    if should_log(args, "debug"):
        print_target_sanity(test_sessions, test, context)
        print_nan_inf_after_preprocessing(("test", test))

    train_loader = engine_trainer.make_loader(
        train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = None
    if val is not None:
        val_loader = engine_trainer.make_loader(
            val,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    test_loader = engine_trainer.make_loader(
        test,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_metrics_history = []
    if args.test_every_epochs > 0:
        console(
            args,
            "target_interval_evaluation=diagnostic_only "
            "target labels are not used for training, checkpoint selection, early stopping, or model selection; "
            f"final checkpoint still follows checkpoint_policy={args.checkpoint_policy}",
            "verbose",
        )

    class_weights = engine_trainer.compute_class_weights(train["y"], args.class_balance)
    if should_log(args, "debug"):
        print_class_balance(train["y"], args.class_balance, class_weights)
    criterion_weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)

    model_config = eegnet_model_config(args, context)
    if should_log(args, "debug"):
        print_model_and_training_config(args, model_config)
    set_seed(args.seed, deterministic=args.deterministic)
    model = build_model(args.model, context.input_channels, context.input_samples, context.num_classes, args).to(device)
    optimizer = engine_trainer.make_optimizer(model, args)
    scheduler = engine_trainer.make_scheduler(optimizer, args)
    criterion = engine_trainer.make_criterion(args, criterion_weight)
    initial_checksum = engine_trainer.model_parameter_checksum(model)
    if args.debug_repro:
        print(f"debug_repro initial_parameter_checksum={initial_checksum}")
        print(f"debug_repro first_20_train_sample_ids={engine_trainer.first_shuffled_sample_ids(train, args.seed, 20)}")

    checkpoint_tracker = engine_checkpointing.CheckpointTracker(
        policy=args.checkpoint_policy,
        monitor_metric=args.monitor_metric,
        min_delta=args.min_delta,
        fixed_eval_epoch=args.fixed_eval_epoch,
    )
    best_target_diagnostic_auc = -np.inf
    best_target_diagnostic_epoch = 0
    best_target_diagnostic_metrics = None
    epoch_rows = []
    if should_log(args, "normal"):
        print_epoch_header(val_loader is not None)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = engine_trainer.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=args.grad_clip_norm,
            show_progress=should_log(args, "debug"),
        )
        if args.debug_repro and epoch == 1:
            print(f"debug_repro epoch1_parameter_checksum={engine_trainer.model_parameter_checksum(model)}")
        val_metrics = None
        monitor_value = np.nan
        if val_loader is not None:
            val_logits, val_y = engine_evaluator.predict_logits(model, val_loader, device)
            val_probs = softmax(val_logits)
            val_pred = val_logits.argmax(axis=1)
            val_metrics = classification_metrics(val_y, val_pred, val_probs[:, 1])
            monitor_value = _monitor_value(val_metrics, args.monitor_metric)
        checkpoint_tracker.update_validation(epoch=epoch, model=model, val_metrics=val_metrics)
        checkpoint_tracker.update_selected(epoch=epoch, model=model)

        if scheduler is not None and val_metrics is not None:
            scheduler.step(monitor_value)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_rows.append(epoch_metrics_row(epoch, train_loss, train_metrics, val_metrics, current_lr, monitor_value))
        print_epoch_row(args, epoch_rows[-1], val_metrics is not None, checkpoint_tracker.epochs_without_improvement)
        if args.test_every_epochs > 0 and epoch % args.test_every_epochs == 0:
            epoch_metrics = evaluate_current_model_on_target(model, test_loader, device)
            test_metrics_history.append(metrics_history_row(epoch, epoch_metrics, reason="interval_diagnostic"))
            epoch_target_auc = _monitor_value(epoch_metrics, "roc_auc")
            if epoch_target_auc > best_target_diagnostic_auc:
                best_target_diagnostic_auc = epoch_target_auc
                best_target_diagnostic_epoch = epoch
                best_target_diagnostic_metrics = epoch_metrics
            if should_log(args, "verbose"):
                print_epoch_test_metrics(plan.target_subject, epoch, epoch_metrics)
        if early_stop_enabled(args) and checkpoint_tracker.epochs_without_improvement >= args.early_stop_patience:
            console(
                args,
                f"early_stopping_triggered epoch={epoch} "
                f"best_epoch={checkpoint_tracker.best_epoch} monitor_metric={args.monitor_metric} "
                f"best_monitor={checkpoint_tracker.best_monitor:.4f}",
                "normal",
            )
            break

    checkpoint_selection = checkpoint_tracker.finalize()
    selected_state = checkpoint_selection["selected_state"]
    selected_epoch = int(checkpoint_selection["selected_epoch"])
    selected_reason = str(checkpoint_selection["selected_reason"])
    best_epoch = int(checkpoint_selection["best_epoch"])
    best_monitor = float(checkpoint_selection["best_monitor"])
    best_macro_f1 = float(checkpoint_selection["best_macro_f1"])
    best_balanced_acc = float(checkpoint_selection["best_balanced_acc"])
    if args.debug_repro:
        print(f"debug_repro best_epoch={best_epoch} best_validation_metric={best_monitor:.10f}")
        print(f"debug_repro selected_epoch={selected_epoch} selected_reason={selected_reason}")
    if best_target_diagnostic_metrics is not None:
        if should_log(args, "verbose"):
            print_best_target_diagnostic_metrics(best_target_diagnostic_epoch, best_target_diagnostic_metrics)
    model.load_state_dict(selected_state)
    method = make_method(args)
    target_unlabeled_loader = engine_trainer.make_unlabeled_loader(test, args.batch_size, args.num_workers, args.seed)
    model = method.adapt(
        model,
        target_unlabeled_loader,
        ctx={
            "fold": plan,
            "adaptation_protocol": args.adaptation_protocol,
            "device": device,
            "log_fn": (lambda message: console(args, message, "normal")),
        },
    )
    method_diagnostics = method.diagnostics()
    if val is not None and plan.outputs_enabled:
        save_validation_subject_metrics(
            model,
            val,
            plan.val_metrics_path,
            device,
            args.batch_size,
            args.num_workers,
            args.seed,
        )

    target_eval = engine_evaluator.evaluate_target(model, test_loader, device)
    test_y = target_eval["y_true"]
    test_probs = target_eval["probs"]
    test_pred = target_eval["y_pred"]
    test_metrics = target_eval["metrics"]
    print_fold_result(args, test_metrics, plan)
    has_validation = val_loader is not None
    if test_metrics_history:
        test_metrics_history.append(metrics_history_row(selected_epoch, test_metrics, reason="final_selected_checkpoint"))
        if plan.outputs_enabled:
            save_test_metrics_history(plan.test_metrics_path, test_metrics_history)

    if plan.outputs_enabled:
        save_predictions(plan.prediction_path, test, test_y, test_pred, test_probs, plan)
        save_predictions(report_prediction_path(plan), test, test_y, test_pred, test_probs, plan)
        method_prediction_path, method_metrics_path = save_method_target_artifacts(
            args,
            plan,
            test,
            test_y,
            test_pred,
            test_probs,
            test_metrics,
        )
        save_epoch_metrics_report(plan, epoch_rows)
    checkpoint = {
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "dataset_path": dataset_path_for_manifest(args),
        "dataset": plan.dataset,
        "dataset_name": plan.dataset,
        "protocol_name": plan.label_protocol,
        "model": plan.model_name,
        "model_name": plan.model_name,
        "method": args.method,
        "adaptation_protocol": args.adaptation_protocol,
        "input_channels": plan.input_channels,
        "input_samples": plan.input_samples,
        "num_classes": plan.num_classes,
        "label_protocol": plan.label_protocol,
        "command": plan.command,
        "reproducibility": repro_metadata,
        "model_state_dict": selected_state,
        "model_config": model_config,
        "training_config": training_config(args),
        "args": vars(args),
        "label_mode": args.label_mode,
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "target_id_space": args.target_id_space,
        "seed": args.seed,
        "class_balance": args.class_balance,
        "loss_type": args.loss_type,
        "train_subject_ids": plan.train_subject_ids,
        "train_subject_raw_ids": plan.train_subject_raw_ids,
        "val_subject_ids": plan.val_subject_ids,
        "val_subject_raw_ids": plan.val_subject_raw_ids,
        "normalization_mean": torch.from_numpy(preprocess_state["mean"].copy()),
        "normalization_std": torch.from_numpy(preprocess_state["std"].copy()),
        "clipping_thresholds": engine_trainer.tensorize_clip_bounds(preprocess_state["clip_bounds"]),
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "best_epoch": None if not has_validation else best_epoch,
        "best_target_diagnostic_epoch": best_target_diagnostic_epoch,
        "best_target_diagnostic_metrics": (
            None if best_target_diagnostic_metrics is None else serializable_metrics(best_target_diagnostic_metrics)
        ),
        "selected_epoch": selected_epoch,
        "selected_reason": selected_reason,
        "validation_mode": args.validation_mode,
        "val_ratio": args.val_ratio,
        "val_subject_ratio": args.val_subject_ratio,
        "checkpoint_policy": args.checkpoint_policy,
        "test_every_epochs": args.test_every_epochs,
        "test_metrics_history_path": str(plan.test_metrics_path) if test_metrics_history else "",
        "early_stop_enabled": early_stop_enabled(args),
        "best_val_metric": {
            "macro_f1": None if not has_validation else best_macro_f1,
            "balanced_accuracy": None if not has_validation else best_balanced_acc,
            args.monitor_metric: None if not has_validation else best_monitor,
        },
        "final_metrics": serializable_metrics(test_metrics),
    }
    adabn_checkpoint_path = adapted_checkpoint_path(plan, "adabn")
    adabn_report = None
    if args.method == "adabn":
        adabn_report = {
            **method_diagnostics,
            "method": "adabn",
            "source_checkpoint_path": str(plan.checkpoint_path),
            "adapted_checkpoint_path": str(adabn_checkpoint_path),
            "target_subject": plan.target_subject,
            "target_subject_raw": plan.target_subject_raw,
            "adaptation_protocol": args.adaptation_protocol,
            "target_adaptation_mode": "target_test_unlabeled",
            "target_labels_used_for_adaptation": False,
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_evaluation_only": True,
        }
    if plan.outputs_enabled:
        plan.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, plan.checkpoint_path)
        source_ckpt_path = save_source_fold_artifacts(
            args,
            plan,
            checkpoint,
            preprocess_state,
            class_weights,
            train["y"],
            epoch_rows,
            best_epoch=None if val_loader is None else best_epoch,
            best_val_metric=np.nan if val_loader is None else best_monitor,
            selected_epoch=selected_epoch,
            selected_reason=selected_reason,
        )
        if args.method == "adabn":
            adabn_report["source_checkpoint_path"] = str(source_ckpt_path)
            adabn_report["adapted_checkpoint_path"] = str(adabn_checkpoint_path) if args.save_adapted_checkpoint else ""
            adapted_checkpoint = {
                **checkpoint,
                "model_state_dict": engine_checkpointing.copy_model_state(model),
                "source_checkpoint_path": str(source_ckpt_path),
                "adapted_checkpoint_path": str(adabn_checkpoint_path),
                "adabn_report": adabn_report,
            }
            if args.save_adapted_checkpoint:
                torch.save(adapted_checkpoint, adabn_checkpoint_path)
            write_json(adabn_report_path(plan), adabn_report)
            write_adaptation_manifest_row(
                adaptation_manifest_path(plan),
                args,
                plan,
                source_checkpoint_path=source_ckpt_path,
                adapted_checkpoint_path_value=adabn_checkpoint_path if args.save_adapted_checkpoint else None,
                prediction_path=method_prediction_path,
                metrics_path=method_metrics_path,
                adaptation_report_path=adabn_report_path(plan),
                target_samples_used=int(adabn_report.get("target_samples_used", 0)),
            )
        if plan.latest_checkpoint_path is not None:
            plan.latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, plan.latest_checkpoint_path)
        save_success_summary_row(
            plan.summary_path,
            args=args,
            plan=plan,
            metrics=test_metrics,
            class_weights=class_weights,
            best_epoch=None if val_loader is None else best_epoch,
            best_macro_f1=np.nan if val_loader is None else best_macro_f1,
            best_balanced_acc=np.nan if val_loader is None else best_balanced_acc,
            best_monitor=np.nan if val_loader is None else best_monitor,
            selected_epoch=selected_epoch,
            selected_reason=selected_reason,
        )
        write_checkpoint_manifest_row(
            plan.manifest_path,
            args,
            plan,
            status="success",
            best_epoch=selected_epoch,
            best_val_metric=best_monitor,
            error="",
        )
        save_fold_reporting(args, plan, test_metrics, selected_epoch, selected_reason)
        fold_audit.update(
            {
                "preprocessing": {
                    "normalization_source": "source_training_only",
                    "robust_clipping": bool(args.robust_clip),
                    "post_preprocess_nan_inf": audit_nan_inf(train=train, val=val, test=test),
                },
                "artifacts": fold_artifacts(plan),
                "selected_epoch": selected_epoch,
                "selected_reason": selected_reason,
                "method_diagnostics": method_diagnostics,
                "confusion_matrix": np.asarray(test_metrics["confusion_matrix"]).tolist(),
                "final_metrics": serializable_metrics(test_metrics),
            }
        )
        if args.method == "adabn" and adabn_report is not None:
            fold_audit.update(
                {
                    "adaptation_protocol": args.adaptation_protocol,
                    "target_unlabeled_used_for_adaptation": True,
                    "target_bn_stats_recomputed_on_target": bool(adabn_report.get("bn_running_stats_changed")),
                    "target_labels_used_for_adaptation": False,
                    "target_labels_used_for_model_selection": False,
                    "target_labels_used_for_evaluation_only": True,
                    "adaptation_eval_split": "same_as_adaptation",
                    "adaptation_report_path": str(adabn_report_path(plan)),
                    "adapted_checkpoint_path": str(adabn_checkpoint_path),
                    "adaptation": adabn_report,
                }
            )
        save_fold_audit(plan, fold_audit)
        update_aggregate_metrics_report(plan)
        update_artifacts_report(plan)
        if should_log(args, "debug"):
            print(f"Saved predictions: {plan.prediction_path}")
            print(f"Saved checkpoint: {plan.checkpoint_path}")
        if plan.latest_checkpoint_path is not None:
            console(args, f"Saved latest checkpoint: {plan.latest_checkpoint_path}", "debug")
        console(args, f"Saved summary: {plan.summary_path}", "debug")
        if val is not None:
            console(args, f"Saved validation metrics: {plan.val_metrics_path}", "debug")
        if test_metrics_history:
            console(args, f"Saved target diagnostic metrics: {plan.test_metrics_path}", "debug")
    else:
        console(args, "Output saving disabled: --output-dir none", "debug")


def run_reuse_source_fold(
    args: argparse.Namespace,
    context: DatasetContext,
    plan: FoldPlan,
    device: torch.device,
    repro_metadata: dict[str, object],
    *,
    fold_index: int,
    fold_total: int,
) -> None:
    if args.method == "source_only":
        raise ValueError("reuse-source mode is only valid for adaptation methods")
    set_seed(args.seed, deterministic=args.deterministic)
    guard_reuse_outputs(args, plan)
    write_fold_report(args, context, plan)
    if should_log(args, "normal"):
        print_compact_fold_start(plan, fold_index, fold_total, dry_run=False)

    source_row = find_source_checkpoint_row(args, context, plan)
    if source_row is None:
        message = f"No source checkpoint found for target_subject={plan.target_subject}"
        if args.require_source_checkpoint:
            raise FileNotFoundError(message)
        console(args, f"[WARN] {message}; skipping fold.", "normal")
        return

    source_checkpoint = Path(str(source_row["checkpoint_path"]))
    normalization_path = Path(str(source_row["normalization_stats_path"]))
    split_path = Path(str(source_row["split_info_path"]))
    class_weight_path = Path(str(source_row["class_weights_path"]))
    if should_log(args, "normal"):
        print(f"Loaded source checkpoint: {source_checkpoint}")
        print(f"Loaded normalization statistics: {normalization_path}")
        print(f"Loaded split info: {split_path}")
        print("")

    test, _test_sessions = load_target_arrays(args, context, plan)
    preprocess_state = load_normalization_stats(normalization_path)
    test["x"] = engine_trainer.preprocess_target(test["x"], preprocess_state)
    test_loader = engine_trainer.make_loader(
        test,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    target_unlabeled_loader = engine_trainer.make_unlabeled_loader(test, args.batch_size, args.num_workers, args.seed)

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") or eegnet_model_config(args, context)
    set_seed(args.seed, deterministic=args.deterministic)
    model = build_model(args.model, context.input_channels, context.input_samples, context.num_classes, args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    method = make_method(args)
    model = method.adapt(
        model,
        target_unlabeled_loader,
        ctx={
            "fold": plan,
            "adaptation_protocol": args.adaptation_protocol,
            "device": device,
            "log_fn": (lambda message: console(args, message, "normal")),
        },
    )
    method_diagnostics = method.diagnostics()
    target_eval = engine_evaluator.evaluate_target(model, test_loader, device)
    test_y = target_eval["y_true"]
    test_probs = target_eval["probs"]
    test_pred = target_eval["y_pred"]
    test_metrics = target_eval["metrics"]
    print_fold_result(args, test_metrics, plan)

    selected_epoch = int(checkpoint.get("selected_epoch", source_row.get("selected_epoch", source_row.get("best_epoch", 0))))
    selected_reason = str(checkpoint.get("selected_reason", source_row.get("selected_reason", "source_manifest")))
    best_epoch = checkpoint.get("best_epoch", source_row.get("best_epoch", selected_epoch))
    best_val_metric = source_row.get("val_metric_value", source_row.get("best_val_metric", np.nan))
    class_weights = load_class_weights_for_reuse(class_weight_path)

    if plan.outputs_enabled:
        save_predictions(plan.prediction_path, test, test_y, test_pred, test_probs, plan)
        save_predictions(report_prediction_path(plan), test, test_y, test_pred, test_probs, plan)
        method_prediction_path, method_metrics_path = save_method_target_artifacts(
            args,
            plan,
            test,
            test_y,
            test_pred,
            test_probs,
            test_metrics,
        )
        fold_dir = fold_artifact_dir(plan)
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "source_checkpoint_used.txt").write_text(str(source_checkpoint) + "\n", encoding="utf-8")
        adabn_ckpt_path = adapted_checkpoint_path(plan, args.method)
        adabn_report = {
            **method_diagnostics,
            "method": args.method,
            "source_checkpoint_path": str(source_checkpoint),
            "adapted_checkpoint_path": str(adabn_ckpt_path) if args.save_adapted_checkpoint else "",
            "target_subject": plan.target_subject,
            "target_subject_raw": plan.target_subject_raw,
            "adaptation_protocol": args.adaptation_protocol,
            "target_adaptation_mode": "target_test_unlabeled",
            "target_labels_used_for_adaptation": False,
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_evaluation_only": True,
        }
        if args.save_adapted_checkpoint:
            adabn_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **checkpoint,
                    "model_state_dict": engine_checkpointing.copy_model_state(model),
                    "source_checkpoint_path": str(source_checkpoint),
                    "adapted_checkpoint_path": str(adabn_ckpt_path),
                    "adabn_report": adabn_report,
                },
                adabn_ckpt_path,
            )
        write_json(adabn_report_path(plan), adabn_report)
        write_json(fold_dir / "adaptation_report.json", adabn_report)
        save_success_summary_row(
            plan.summary_path,
            args=args,
            plan=plan,
            metrics=test_metrics,
            class_weights=class_weights,
            best_epoch=None if is_nan_like(best_epoch) else int(float(best_epoch)),
            best_macro_f1=np.nan,
            best_balanced_acc=np.nan,
            best_monitor=float(best_val_metric) if not is_nan_like(best_val_metric) else np.nan,
            selected_epoch=selected_epoch,
            selected_reason=selected_reason,
        )
        write_checkpoint_manifest_row(
            plan.manifest_path,
            args,
            plan,
            status="success",
            best_epoch=selected_epoch,
            best_val_metric=float(best_val_metric) if not is_nan_like(best_val_metric) else np.nan,
            error="",
        )
        write_adaptation_manifest_row(
            adaptation_manifest_path(plan),
            args,
            plan,
            source_checkpoint_path=source_checkpoint,
            adapted_checkpoint_path_value=adabn_ckpt_path if args.save_adapted_checkpoint else None,
            prediction_path=method_prediction_path,
            metrics_path=method_metrics_path,
            adaptation_report_path=adabn_report_path(plan),
            target_samples_used=int(adabn_report.get("target_samples_used", 0)),
        )
        save_fold_reporting(args, plan, test_metrics, selected_epoch, selected_reason)
        fold_audit = {
            "target_subject": plan.target_subject,
            "target_subject_raw": plan.target_subject_raw,
            "train_subject_ids": plan.train_subject_ids,
            "train_subject_raw_ids": plan.train_subject_raw_ids,
            "val_subject_ids": plan.val_subject_ids,
            "val_subject_raw_ids": plan.val_subject_raw_ids,
            "test_subject_ids": plan.test_subject_ids,
            "test_subject_raw_ids": plan.test_subject_raw_ids,
            "validation_mode": args.validation_mode,
            "checkpoint_policy": args.checkpoint_policy,
            "counts": {"train": plan.train_counts, "val": plan.val_counts, "test": plan.test_counts},
            "reuse_source": True,
            "source_checkpoint_path": str(source_checkpoint),
            "normalization_stats_path": str(normalization_path),
            "split_info_path": str(split_path),
            "class_weights_path": str(class_weight_path),
            "preprocessing": {
                "normalization_source": "loaded_source_training_only",
                "robust_clipping": bool(preprocess_state["clip_bounds"] is not None),
                "post_preprocess_nan_inf": audit_nan_inf(test=test),
            },
            "artifacts": fold_artifacts(plan),
            "selected_epoch": selected_epoch,
            "selected_reason": selected_reason,
            "method_diagnostics": method_diagnostics,
            "confusion_matrix": np.asarray(test_metrics["confusion_matrix"]).tolist(),
            "final_metrics": serializable_metrics(test_metrics),
            "adaptation_protocol": args.adaptation_protocol,
            "target_unlabeled_used_for_adaptation": True,
            "target_bn_stats_recomputed_on_target": bool(adabn_report.get("bn_running_stats_changed")),
            "target_labels_used_for_adaptation": False,
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_evaluation_only": True,
            "adaptation_eval_split": "same_as_adaptation",
            "adaptation_report_path": str(adabn_report_path(plan)),
            "adapted_checkpoint_path": str(adabn_ckpt_path) if args.save_adapted_checkpoint else "",
            "adaptation": adabn_report,
        }
        save_fold_audit(plan, fold_audit)
        update_aggregate_metrics_report(plan)
        update_artifacts_report(plan)


def write_fold_report(args: argparse.Namespace, context: DatasetContext, plan: FoldPlan) -> None:
    if not plan.outputs_enabled:
        return
    if context.dataset == "seedvig":
        if context.integrity_report is None:
            raise ValueError("SEED-VIG fold report requires an integrity report")
        write_loso_fold_integrity_report(
            context.integrity_report,
            plan.fold_report_path,
            target_subject=plan.target_subject,
            train_pairs=plan.train_pairs,
            val_pairs=plan.val_pairs,
            test_pairs=plan.test_pairs,
            robust_clip=args.robust_clip,
            validation_mode=args.validation_mode,
            validation_strategy=plan.validation_strategy,
            val_ratio=args.val_ratio,
            val_subject_ratio=args.val_subject_ratio,
            checkpoint_policy=args.checkpoint_policy,
            early_stop_enabled=early_stop_enabled(args),
            train_counts=plan.train_counts,
            val_counts=plan.val_counts,
            test_counts=plan.test_counts,
        )
        return
    lines = [
        "SADT LOSO Fold Integrity Report",
        "",
        f"Dataset: {context.dataset}",
        f"Model: {context.model_name}",
        f"Input channels: {context.input_channels}",
        f"Input samples: {context.input_samples}",
        f"Num classes: {context.num_classes}",
        f"Label protocol: {context.label_protocol}",
        f"Target fold subject ID: {plan.target_subject}",
        f"Target raw subject ID: {plan.target_subject_raw}",
        f"Train subject IDs: {plan.train_subject_ids}",
        f"Train raw subject IDs: {plan.train_subject_raw_ids}",
        f"Validation subject IDs: {plan.val_subject_ids}",
        f"Validation raw subject IDs: {plan.val_subject_raw_ids}",
        f"Test subject IDs: {plan.test_subject_ids}",
        f"Test raw subject IDs: {plan.test_subject_raw_ids}",
        f"Validation mode: {args.validation_mode}",
        f"Validation strategy: {plan.validation_strategy}",
        f"Checkpoint policy: {args.checkpoint_policy}",
        "",
        f"train: samples={plan.train_counts['usable']} alert={plan.train_counts['alert']} fatigue={plan.train_counts['fatigue']}",
        f"val: samples={plan.val_counts['usable']} alert={plan.val_counts['alert']} fatigue={plan.val_counts['fatigue']}",
        f"test: samples={plan.test_counts['usable']} alert={plan.test_counts['alert']} fatigue={plan.test_counts['fatigue']}",
        "",
        "Preprocessing leakage checks:",
        "Normalization statistics source: source-training data only",
        f"Clipping statistics source: {'source-training data only' if args.robust_clip else 'not computed; robust clipping disabled'}",
        "Target labels are audit/evaluation only.",
        "No target samples were used for training, validation, preprocessing statistics, class weights, or model selection.",
    ]
    plan.fold_report_path.parent.mkdir(parents=True, exist_ok=True)
    plan.fold_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_dataset_fold_summary(args: argparse.Namespace, context: DatasetContext, plan: FoldPlan) -> None:
    if context.dataset == "seedvig" and context.integrity_report is not None:
        print_integrity_report_summary(
            context.integrity_report,
            target_subject=plan.target_subject,
            train_pairs=plan.train_pairs,
            val_pairs=plan.val_pairs,
            test_pairs=plan.test_pairs,
            train_counts=plan.train_counts,
            val_counts=plan.val_counts,
            test_counts=plan.test_counts,
        )
    print("dataset_fold_summary")
    print(f"  dataset={context.dataset}")
    print(f"  model={context.model_name}")
    print(f"  input_channels={context.input_channels}")
    print(f"  input_samples={context.input_samples}")
    print(f"  num_classes={context.num_classes}")
    print(f"  label_protocol={context.label_protocol}")
    if context.dataset != "seedvig":
        print(f"  label_mode={args.label_mode} label_mode_applicable=False")
    print(f"  normalization_stats=source_training_only")


def load_fold_arrays(
    args: argparse.Namespace,
    context: DatasetContext,
    plan: FoldPlan,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, np.ndarray], list, list, list]:
    if context.dataset == "seedvig":
        train_sessions = load_seedvig_file_pairs(plan.train_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
        if args.validation_mode == "subject_split":
            val_sessions = load_seedvig_file_pairs(plan.val_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
            train = sessions_to_arrays(train_sessions)
            val = sessions_to_arrays(val_sessions)
        elif args.validation_mode == "sample_stratified":
            val_sessions = []
            source = sessions_to_arrays(train_sessions)
            train, val = split_arrays_stratified(source, val_ratio=args.val_ratio, seed=args.seed)
        elif args.validation_mode == "none":
            val_sessions = []
            train = sessions_to_arrays(train_sessions)
            val = None
        else:
            raise ValueError(f"Unsupported validation mode: {args.validation_mode}")
        test_sessions = load_seedvig_file_pairs(plan.test_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
        test = sessions_to_arrays(test_sessions)
        return train, val, test, train_sessions, val_sessions, test_sessions

    if context.sadt_arrays is None:
        raise ValueError("SADT arrays are not loaded")
    source = subset_by_subject(context.sadt_arrays, plan.target_subject, include=False)
    test = subset_by_subject(context.sadt_arrays, plan.target_subject, include=True)
    if args.validation_mode == "subject_split":
        train = subset_by_subject_ids(source, plan.train_subject_ids)
        val = subset_by_subject_ids(source, plan.val_subject_ids)
    elif args.validation_mode == "sample_stratified":
        train, val = split_arrays_stratified(source, val_ratio=args.val_ratio, seed=args.seed)
    elif args.validation_mode == "none":
        train = source
        val = None
    else:
        raise ValueError(f"Unsupported validation mode: {args.validation_mode}")
    return train, val, test, [], [], []


def load_target_arrays(
    args: argparse.Namespace,
    context: DatasetContext,
    plan: FoldPlan,
) -> tuple[dict[str, np.ndarray], list]:
    if context.dataset == "seedvig":
        test_sessions = load_seedvig_file_pairs(plan.test_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
        return sessions_to_arrays(test_sessions), test_sessions
    if context.sadt_arrays is None:
        raise ValueError("Array dataset is not loaded")
    return subset_by_subject(context.sadt_arrays, plan.target_subject, include=True), []


def resolve_source_manifest_path(args: argparse.Namespace) -> Path:
    if args.source_manifest is not None:
        return Path(args.source_manifest)
    if args.source_checkpoint_dir is None:
        raise ValueError("--source-manifest or --source-checkpoint-dir is required in reuse-source mode")
    root = Path(args.source_checkpoint_dir)
    candidates = [
        root / "checkpoint_manifest.csv",
        root / "checkpoints" / "checkpoints_manifest.csv",
        root / "checkpoints_manifest.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint manifest found under {root}")


def find_source_checkpoint_row(
    args: argparse.Namespace,
    context: DatasetContext,
    plan: FoldPlan,
) -> pd.Series | None:
    manifest_path = resolve_source_manifest_path(args)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Source manifest is empty: {manifest_path}")
    mask = pd.Series([True] * len(manifest))
    if "method" in manifest.columns:
        mask &= manifest["method"].astype(str) == "source_only"
    model_col = "model_name" if "model_name" in manifest.columns else "model"
    if model_col in manifest.columns:
        mask &= manifest[model_col].astype(str) == str(plan.model_name)
    if "seed" in manifest.columns:
        mask &= pd.to_numeric(manifest["seed"], errors="coerce") == int(args.seed)
    if "target_subject" in manifest.columns:
        mask &= pd.to_numeric(manifest["target_subject"], errors="coerce") == int(plan.target_subject)
    protocol_cols = [col for col in ("protocol_name", "label_protocol") if col in manifest.columns]
    if protocol_cols:
        protocol_mask = pd.Series([False] * len(manifest))
        for col in protocol_cols:
            protocol_mask |= manifest[col].astype(str) == str(plan.label_protocol)
        mask &= protocol_mask
    rows = manifest.loc[mask].copy()
    if rows.empty:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"Multiple source checkpoints match target_subject={plan.target_subject} in {manifest_path}. "
            "Use a manifest with unique rows for this benchmark run."
        )
    row = rows.iloc[0]
    required = ["checkpoint_path", "normalization_stats_path", "split_info_path", "class_weights_path"]
    missing = [col for col in required if col not in row.index or pd.isna(row[col]) or str(row[col]) == ""]
    if missing:
        raise ValueError(f"Source manifest row is missing required columns: {missing}")
    for col in required:
        if not Path(str(row[col])).exists():
            raise FileNotFoundError(f"Manifest column {col} points to a missing file: {row[col]}")
    return row


def load_class_weights_for_reuse(path: str | Path) -> np.ndarray | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("class_weights")
    if weights is None:
        return None
    return np.asarray(weights, dtype=np.float32)


def is_nan_like(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def reproducibility_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else ""
    git_commit = get_git_commit_hash()
    return {
        "seed": args.seed,
        "deterministic": args.deterministic,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": cuda_device_name,
        "device": str(device),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "num_workers": args.num_workers,
        "train_dataloader_shuffle": True,
        "val_dataloader_shuffle": False,
        "test_dataloader_shuffle": False,
        "dataloader_generator_seed": args.seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "git_commit": git_commit,
    }


def print_reproducibility_metadata(metadata: dict[str, object]) -> None:
    print("reproducibility")
    for key, value in metadata.items():
        print(f"  {key}={value}")


def run_dir(plan: FoldPlan) -> Path:
    if plan.summary_path.parent.name == "summaries":
        return plan.summary_path.parent.parent
    return plan.summary_path.parent


def report_prediction_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "predictions" / f"fold_{plan.target_subject}.csv"


def fold_artifact_dir(plan: FoldPlan) -> Path:
    return run_dir(plan) / f"fold_subject_{plan.target_subject}"


def fold_source_checkpoint_path(plan: FoldPlan) -> Path:
    return fold_artifact_dir(plan) / "source_checkpoint.pt"


def fold_config_path(plan: FoldPlan) -> Path:
    return fold_artifact_dir(plan) / "fold_config.json"


def split_info_path(plan: FoldPlan) -> Path:
    return fold_artifact_dir(plan) / "split_info.json"


def normalization_stats_path(plan: FoldPlan) -> Path:
    return fold_artifact_dir(plan) / "normalization_stats.npz"


def class_weights_path(plan: FoldPlan) -> Path:
    return fold_artifact_dir(plan) / "class_weights.json"


def train_log_path(plan: FoldPlan) -> Path:
    return fold_artifact_dir(plan) / "train_log.csv"


def fold_target_predictions_path(plan: FoldPlan, method_name: str) -> Path:
    return fold_artifact_dir(plan) / f"target_predictions_{method_name}.csv"


def fold_target_metrics_path(plan: FoldPlan, method_name: str) -> Path:
    return fold_artifact_dir(plan) / f"target_metrics_{method_name}.json"


def source_checkpoint_manifest_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "checkpoint_manifest.csv"


def adaptation_manifest_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "adaptation_manifest.csv"


def epoch_metrics_report_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "metrics" / f"epoch_metrics_fold_{plan.target_subject}.csv"


def fold_metrics_report_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "metrics" / "fold_metrics.csv"


def aggregate_metrics_report_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "metrics" / "aggregate_metrics.json"


def split_audit_report_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "split_audit" / f"fold_{plan.target_subject}.json"


def adabn_report_path(plan: FoldPlan) -> Path:
    return run_dir(plan) / "reports" / f"adabn_report_subject_{plan.target_subject}.json"


def adapted_checkpoint_path(plan: FoldPlan, method_name: str) -> Path:
    return plan.checkpoint_path.with_name(plan.checkpoint_path.stem + f"_{method_name}_adapted.pt")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_normalization_stats(path: Path, preprocess_state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clip_bounds = preprocess_state.get("clip_bounds")
    payload = {
        "mean": np.asarray(preprocess_state["mean"], dtype=np.float32),
        "std": np.asarray(preprocess_state["std"], dtype=np.float32),
        "robust_clip": np.asarray(clip_bounds is not None),
    }
    if clip_bounds is not None:
        lo, hi = clip_bounds
        payload["clip_low"] = np.asarray(lo, dtype=np.float32)
        payload["clip_high"] = np.asarray(hi, dtype=np.float32)
    np.savez(path, **payload)


def load_normalization_stats(path: str | Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        clip_bounds = None
        if bool(np.asarray(data.get("robust_clip", False)).item()):
            clip_bounds = (np.asarray(data["clip_low"], dtype=np.float32), np.asarray(data["clip_high"], dtype=np.float32))
        return {
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "std": np.asarray(data["std"], dtype=np.float32),
            "clip_bounds": clip_bounds,
        }


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_run_reports(
    args: argparse.Namespace,
    context: DatasetContext,
    plans: list[FoldPlan],
    target_subjects: list[int],
    repro_metadata: dict[str, object],
) -> None:
    if not plans or not outputs_enabled(args):
        return
    root = run_dir(plans[0])
    write_json(
        root / "run_config.json",
        {
            "cli_args": vars(args),
            "dataset": context.dataset,
            "model": context.model_name,
            "method": args.method,
            "adaptation_protocol": args.adaptation_protocol,
            "reuse_source": args.reuse_source,
            "source_manifest": args.source_manifest,
            "source_checkpoint_dir": args.source_checkpoint_dir,
            "require_source_checkpoint": args.require_source_checkpoint,
            "save_adapted_checkpoint": args.save_adapted_checkpoint,
            "adabn": {
                "reset_stats": args.adabn_reset_stats,
                "momentum": args.adabn_momentum,
                "num_passes": args.adabn_num_passes,
            },
            "label_protocol": context.label_protocol,
            "input_channels": context.input_channels,
            "input_samples": context.input_samples,
            "num_classes": context.num_classes,
            "subjects": context.subjects,
            "subject_mapping": context.subject_mapping,
            "target_id_space": args.target_id_space,
            "selected_targets": target_subjects,
            "selected_target_raw_ids": raw_subject_ids(context, target_subjects),
            "selected_targets_display": format_selected_targets(context, target_subjects),
            "label_config": {
                "label_mode": args.label_mode,
                "label_mode_explicit": bool(getattr(args, "label_mode_explicit", False)),
                "label_mode_applicable": context.dataset == "seedvig",
                "label_protocol": context.label_protocol,
            },
        },
    )
    write_json(root / "reproducibility.json", repro_metadata)
    write_json(
        root / "model_selection_policy.json",
        {
            "validation_mode": args.validation_mode,
            "checkpoint_policy": args.checkpoint_policy,
            "method": args.method,
            "adaptation_protocol": args.adaptation_protocol,
            "reuse_source": args.reuse_source,
            "source_manifest": args.source_manifest,
            "source_checkpoint_dir": args.source_checkpoint_dir,
            "early_stop_enabled": early_stop_enabled(args),
            "monitor_metric": args.monitor_metric,
            "target_labels_for_model_selection": False,
            "target_unlabeled_used_for_adaptation": args.method == "adabn",
            "target_bn_stats_recomputed_on_target": args.method == "adabn",
            "target_labels_used_for_adaptation": False,
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_evaluation_only": True,
            "normalization_source": "source_training_only",
            "target_interval_diagnostics": "audit_only" if args.test_every_epochs > 0 else "disabled",
        },
    )


def print_run_overview(
    args: argparse.Namespace,
    context: DatasetContext,
    target_subjects: list[int],
    plans: list[FoldPlan],
    device: torch.device,
) -> None:
    output_dir = "none" if not plans else str(run_dir(plans[0])) if outputs_enabled(args) else "none"
    print("EEGDA Benchmark Run")
    print("-" * 21)
    print(f"Dataset        : {context.dataset}")
    method_label = "source-only" if args.method == "source_only" else "AdaBN source-free"
    print(f"Protocol       : LOSO {method_label}")
    print(f"Model          : {'EEGNet' if context.model_name == 'eegnet' else context.model_name}")
    print(f"Subjects       : {len(context.subjects)} available, {len(target_subjects)} selected")
    print(f"Selected       : {format_selected_targets(context, target_subjects)}")
    print(f"Input shape    : {context.input_channels} channels x {context.input_samples} samples")
    print(f"Classes        : {context.num_classes}")
    print(f"Label protocol : {context.label_protocol}")
    print(f"Class balance  : {loss_label(args)}")
    print(f"Validation     : {args.validation_mode}")
    print(f"Checkpoint     : {checkpoint_label(args)}")
    if args.reuse_source:
        print("Source model   : reused from manifest")
    print(f"Device         : {device}")
    print(f"Seed           : {args.seed}")
    print(f"Output dir     : {output_dir}")
    print("")
    print("Protocol checks")
    print("[OK] Target labels are not used for model selection.")
    print("[OK] Normalization statistics are computed from source training data only.")
    if args.method == "adabn":
        print("[OK] AdaBN uses unlabeled target data only and updates BatchNorm running statistics only.")
        if args.reuse_source:
            print("[OK] Source training is skipped; fold source checkpoints are loaded from the source manifest.")
    if args.validation_mode == "none":
        print(f"[WARN] No validation set is used; checkpoint_policy={args.checkpoint_policy} will be used.")
    if context.dataset != "seedvig" and getattr(args, "label_mode_explicit", False):
        print(f"[WARN] label_mode={args.label_mode} was provided but is not applicable to this label protocol.")
    if not args.run_all_loso and len(target_subjects) < len(context.subjects):
        print(f"[WARN] Partial LOSO run: {len(target_subjects)} / {len(context.subjects)} subjects selected.")
    print("")


def format_selected_targets(context: DatasetContext, target_subjects: list[int]) -> str:
    return ", ".join(
        f"{subject}(raw={context.subject_mapping.get(subject, subject)})"
        for subject in target_subjects
    )


def loss_label(args: argparse.Namespace) -> str:
    if args.loss_type == "weighted_ce" or args.class_balance == "weighted_loss":
        return "weighted CE"
    return args.loss_type


def checkpoint_label(args: argparse.Namespace) -> str:
    if args.checkpoint_policy == "last":
        return "last epoch"
    if args.checkpoint_policy == "best_val":
        return f"best validation {args.monitor_metric}"
    if args.checkpoint_policy == "fixed_epoch":
        return f"fixed epoch {args.fixed_eval_epoch}"
    return args.checkpoint_policy


def print_compact_fold_start(plan: FoldPlan, fold_index: int, fold_total: int, *, dry_run: bool) -> None:
    prefix = "Dry-run fold" if dry_run else "Fold"
    print(f"{prefix} {fold_index}/{fold_total} | target subject: {plan.target_subject} (raw id: {plan.target_subject_raw})")
    print(
        f"Train: {plan.train_counts['usable']} samples | "
        f"alert={plan.train_counts['alert']} | fatigue={plan.train_counts['fatigue']}"
    )
    if plan.val_counts["usable"] > 0:
        print(
            f"Val  : {plan.val_counts['usable']} samples | "
            f"alert={plan.val_counts['alert']} | fatigue={plan.val_counts['fatigue']}"
        )
    print(f"Test : {plan.test_counts['usable']} samples | labels hidden during training/adaptation")
    print("")


def print_epoch_header(has_validation: bool) -> None:
    if has_validation:
        print("Epoch | loss   | train_auc | val_macro_f1 | val_auc | lr       | checkpoint")
    else:
        print("Epoch | loss   | train_auc | train_macro_f1 | lr")


def epoch_metrics_row(
    epoch: int,
    train_loss: float,
    train_metrics: dict[str, object],
    val_metrics: dict[str, object] | None,
    lr: float,
    monitor_value: float,
) -> dict[str, object]:
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_roc_auc": train_metrics.get("roc_auc"),
        "train_macro_f1": train_metrics.get("macro_f1"),
        "lr": lr,
    }
    if val_metrics is not None:
        row.update(
            {
                "val_macro_f1": val_metrics.get("macro_f1"),
                "val_roc_auc": val_metrics.get("roc_auc"),
                "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
                "monitor_value": monitor_value,
            }
        )
    return row


def print_epoch_row(args: argparse.Namespace, row: dict[str, object], has_validation: bool, no_improve: int) -> None:
    if not should_log(args, "normal"):
        return
    epoch = int(row["epoch"])
    if not should_print_epoch(args, epoch):
        return
    if has_validation:
        checkpoint = "hold" if no_improve else "best"
        print(
            f"{epoch:03d}   | {float(row['train_loss']):.4f} | "
            f"{format_metric(row['train_roc_auc'])}    | {format_metric(row['val_macro_f1'])}        | "
            f"{format_metric(row['val_roc_auc'])} | {float(row['lr']):.2e} | {checkpoint}"
        )
    else:
        print(
            f"{epoch:03d}   | {float(row['train_loss']):.4f} | "
            f"{format_metric(row['train_roc_auc'])}    | {format_metric(row['train_macro_f1'])}         | "
            f"{float(row['lr']):.2e}"
        )


def should_print_epoch(args: argparse.Namespace, epoch: int) -> bool:
    return epoch == 1 or epoch == args.epochs or epoch % args.epoch_log_interval == 0


def get_git_commit_hash() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""
    return ""


def split_loso_file_pairs(file_pairs, *, target_subject: int, val_subject_ratio: float, seed: int):
    test_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) == target_subject]
    source_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) != target_subject]
    source_subjects = sorted({parse_subject_id(raw) for raw, _ in source_pairs})
    rng = np.random.default_rng(seed)
    shuffled = np.array(source_subjects)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(source_subjects) * val_subject_ratio)))
    val_subjects = set(int(s) for s in shuffled[:val_count])

    val_pairs = [(raw, label) for raw, label in source_pairs if parse_subject_id(raw) in val_subjects]
    train_pairs = [(raw, label) for raw, label in source_pairs if parse_subject_id(raw) not in val_subjects]
    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError("Invalid LOSO split produced an empty train/val/test partition")
    return train_pairs, val_pairs, test_pairs


def split_subject_ids(subjects: list[int], val_subject_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(subjects))
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(subjects) * val_subject_ratio)))
    val_subjects = sorted(int(subject) for subject in shuffled[:val_count])
    train_subjects = sorted(int(subject) for subject in shuffled[val_count:])
    if not train_subjects or not val_subjects:
        raise ValueError("Subject split produced an empty train or validation partition")
    return train_subjects, val_subjects


def subset_by_subject(arrays: dict[str, np.ndarray], subject_id: int, *, include: bool) -> dict[str, np.ndarray]:
    mask = arrays["subject_id"] == subject_id
    if not include:
        mask = ~mask
    return index_arrays(arrays, np.flatnonzero(mask))


def subset_by_subject_ids(arrays: dict[str, np.ndarray], subject_ids: list[int]) -> dict[str, np.ndarray]:
    mask = np.isin(arrays["subject_id"], np.array(subject_ids, dtype=np.int64))
    return index_arrays(arrays, np.flatnonzero(mask))


def split_arrays_stratified(
    arrays: dict[str, np.ndarray],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = arrays["y"].astype(np.int64, copy=False)
    train_indices = []
    val_indices = []
    for label in (0, 1):
        label_indices = np.flatnonzero(y == label)
        if len(label_indices) == 0:
            raise ValueError(f"Cannot stratify validation split because class {label} has no source samples")
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        if val_count >= len(shuffled):
            raise ValueError(f"Validation split would consume all class {label} samples")
        val_indices.append(shuffled[:val_count])
        train_indices.append(shuffled[val_count:])
    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return index_arrays(arrays, train_idx), index_arrays(arrays, val_idx)


def index_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in arrays.items()}


def eegnet_model_config(args: argparse.Namespace, context: DatasetContext) -> dict[str, object]:
    return {
        "channels": context.input_channels,
        "samples": context.input_samples,
        "num_classes": context.num_classes,
        "F1": args.eegnet_f1,
        "D": args.eegnet_d,
        "F2": None if args.eegnet_f2 == 0 else args.eegnet_f2,
        "kernLength": args.eegnet_temporal_kernel,
        "separable_kernel_length": args.eegnet_separable_kernel,
        "dropoutRate": args.eegnet_dropout,
        "norm_rate": args.eegnet_norm_rate,
    }


def training_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "optimizer": args.optimizer,
        "loss_type": args.loss_type,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_scheduler": args.lr_scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "min_lr": args.min_lr,
        "early_stop_patience": args.early_stop_patience,
        "disable_early_stop": args.disable_early_stop,
        "early_stop_enabled": early_stop_enabled(args),
        "validation_mode": args.validation_mode,
        "val_ratio": args.val_ratio,
        "val_subject_ratio": args.val_subject_ratio,
        "checkpoint_policy": args.checkpoint_policy,
        "fixed_eval_epoch": args.fixed_eval_epoch,
        "test_every_epochs": args.test_every_epochs,
        "method": args.method,
        "adaptation_protocol": args.adaptation_protocol,
        "adabn_reset_stats": args.adabn_reset_stats,
        "adabn_momentum": args.adabn_momentum,
        "adabn_num_passes": args.adabn_num_passes,
        "min_delta": args.min_delta,
        "monitor_metric": args.monitor_metric,
    }


def early_stop_enabled(args: argparse.Namespace) -> bool:
    return args.validation_mode != "none" and not args.disable_early_stop and args.early_stop_patience > 0


def print_model_and_training_config(args: argparse.Namespace, model_config: dict[str, object]) -> None:
    print("model_config")
    for key, value in model_config.items():
        print(f"  {key}={value}")
    print("training_config")
    for key, value in training_config(args).items():
        print(f"  {key}={value}")


def print_class_balance(y: np.ndarray, class_balance: str, class_weights: np.ndarray | None) -> None:
    counts = np.bincount(y.astype(np.int64), minlength=2)
    print("class_balance")
    print(f"  mode={class_balance}")
    print(f"  source_train_counts={{0: {int(counts[0])}, 1: {int(counts[1])}}}")
    if class_weights is None:
        print("  class_weights=None")
    else:
        print(f"  class_weights={{0: {class_weights[0]:.6f}, 1: {class_weights[1]:.6f}}}")


def evaluate_current_model_on_target(model, loader, device: torch.device) -> dict[str, object]:
    return engine_evaluator.evaluate_target_metrics(model, loader, device)


def metrics_history_row(epoch: int, metrics: dict[str, object], *, reason: str) -> dict[str, object]:
    return {
        "epoch": epoch,
        "reason": reason,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "fatigue_recall": metrics["fatigue_recall"],
        "sensitivity": metrics["sensitivity"],
        "alert_recall": metrics["alert_recall"],
        "specificity": metrics["specificity"],
        "miss_rate": metrics["miss_rate"],
        "roc_auc": metrics["roc_auc"],
        "auprc": metrics["auprc"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    }


def save_test_metrics_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _monitor_value(metrics: dict[str, object], metric_name: str) -> float:
    value = float(metrics[metric_name])
    if np.isnan(value):
        return -np.inf
    return value


def save_validation_subject_metrics(
    model,
    val: dict[str, np.ndarray],
    path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> None:
    rows = []
    for subject_id in sorted({int(subject) for subject in val["subject_id"]}):
        mask = val["subject_id"] == subject_id
        subject_arrays = {
            "x": np.ascontiguousarray(val["x"][mask]),
            "y": np.ascontiguousarray(val["y"][mask]),
        }
        loader = engine_trainer.make_loader(subject_arrays, batch_size, shuffle=False, num_workers=num_workers, seed=seed)
        logits, y_true = engine_evaluator.predict_logits(model, loader, device)
        probs = softmax(logits)
        y_pred = probs.argmax(axis=1)
        metrics = classification_metrics(y_true, y_pred, probs[:, 1])
        rows.append(
            {
                "val_subject": subject_id,
                "val_subject_raw": first_raw_subject(val, mask),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "fatigue_recall": metrics["fatigue_recall"],
                "alert_recall": metrics["alert_recall"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_predictions(path: Path, arrays: dict[str, np.ndarray], y_true, y_pred, probs, plan: FoldPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    confidence = probs.max(axis=1)
    entropy = entropy_from_probs(probs)
    df = pd.DataFrame(
        {
            "dataset": plan.dataset,
            "model": plan.model_name,
            "sample_id": arrays["sample_id"],
            "subject_id": arrays["subject_id"],
            "subject_id_raw": arrays.get("subject_id_raw", arrays["subject_id"]),
            "session_id": arrays["session_id"],
            "file_name": arrays["file_name"],
            "window_id": arrays["window_id"],
            "perclos_value": arrays["perclos_value"],
            "label": arrays["y"],
            "label_mode": arrays["label_mode"],
            "is_valid_binary_sample": arrays["is_valid_binary_sample"],
            "y_true": y_true,
            "y_pred": y_pred,
            "p_0": probs[:, 0],
            "p_1": probs[:, 1],
            "confidence": confidence,
            "entropy": entropy,
        }
    )
    df.to_csv(path, index=False)


def save_source_fold_artifacts(
    args: argparse.Namespace,
    plan: FoldPlan,
    checkpoint: dict[str, object],
    preprocess_state: dict[str, object],
    class_weights: np.ndarray | None,
    train_y: np.ndarray,
    epoch_rows: list[dict[str, object]],
    *,
    best_epoch: int | None,
    best_val_metric: float | None,
    selected_epoch: int,
    selected_reason: str,
) -> Path:
    fold_dir = fold_artifact_dir(plan)
    fold_dir.mkdir(parents=True, exist_ok=True)
    source_ckpt = fold_source_checkpoint_path(plan)
    torch.save(checkpoint, source_ckpt)
    write_json(
        fold_config_path(plan),
        {
            "run_id": plan.run_id,
            "dataset": plan.dataset,
            "model": plan.model_name,
            "method": "source_only",
            "label_protocol": plan.label_protocol,
            "target_subject": plan.target_subject,
            "target_subject_raw": plan.target_subject_raw,
            "seed": args.seed,
            "validation_mode": args.validation_mode,
            "checkpoint_policy": args.checkpoint_policy,
            "monitor_metric": args.monitor_metric,
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "selected_epoch": selected_epoch,
            "selected_reason": selected_reason,
        },
    )
    write_json(
        split_info_path(plan),
        {
            "target_subject": plan.target_subject,
            "target_subject_raw": plan.target_subject_raw,
            "target_id_space": args.target_id_space,
            "train_subjects": plan.train_subject_ids,
            "train_subject_raw_ids": plan.train_subject_raw_ids,
            "val_subjects": plan.val_subject_ids,
            "val_subject_raw_ids": plan.val_subject_raw_ids,
            "test_subjects": plan.test_subject_ids,
            "test_subject_raw_ids": plan.test_subject_raw_ids,
            "train_counts": plan.train_counts,
            "val_counts": plan.val_counts,
            "test_counts": plan.test_counts,
            "validation_mode": args.validation_mode,
            "validation_strategy": plan.validation_strategy,
        },
    )
    save_normalization_stats(normalization_stats_path(plan), preprocess_state)
    write_json(
        class_weights_path(plan),
        {
            "class_balance": args.class_balance,
            "class_weights": None if class_weights is None else class_weights.tolist(),
            "source_train_counts": np.bincount(np.asarray(train_y, dtype=np.int64), minlength=2).tolist(),
        },
    )
    pd.DataFrame(epoch_rows).to_csv(train_log_path(plan), index=False)
    write_source_checkpoint_manifest_row(
        source_checkpoint_manifest_path(plan),
        args,
        plan,
        checkpoint_path=source_ckpt,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        selected_epoch=selected_epoch,
        selected_reason=selected_reason,
    )
    return source_ckpt


def save_method_target_artifacts(
    args: argparse.Namespace,
    plan: FoldPlan,
    test: dict[str, np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    metrics: dict[str, object],
) -> tuple[Path, Path]:
    prediction_path = fold_target_predictions_path(plan, args.method)
    metrics_path = fold_target_metrics_path(plan, args.method)
    save_predictions(prediction_path, test, y_true, y_pred, probs, plan)
    write_json(metrics_path, serializable_metrics(metrics))
    return prediction_path, metrics_path


def first_raw_subject(arrays: dict[str, np.ndarray], mask: np.ndarray) -> object:
    raw_subjects = arrays.get("subject_id_raw", arrays["subject_id"])
    return python_scalar(raw_subjects[np.flatnonzero(mask)[0]])


def guard_run_outputs(args: argparse.Namespace, plan: FoldPlan) -> None:
    if not plan.outputs_enabled:
        return
    if plan.checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint already exists: {plan.checkpoint_path}. Use --overwrite to replace it.")
    if manifest_has_run_id(plan.manifest_path, plan.run_id) and not args.overwrite:
        raise FileExistsError(f"Run ID already exists in manifest: {plan.run_id}. Use --overwrite to replace it.")


def guard_reuse_outputs(args: argparse.Namespace, plan: FoldPlan) -> None:
    if not plan.outputs_enabled:
        return
    outputs = [plan.prediction_path, fold_target_metrics_path(plan, args.method), fold_target_predictions_path(plan, args.method)]
    if args.save_adapted_checkpoint:
        outputs.append(adapted_checkpoint_path(plan, args.method))
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Reuse-source output already exists: {existing[0]}. Use --overwrite to replace it.")


def manifest_has_run_id(path: Path, run_id: str) -> bool:
    if not path.exists():
        return False
    manifest = pd.read_csv(path)
    return "run_id" in manifest.columns and bool((manifest["run_id"].astype(str) == str(run_id)).any())


def serializable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def initial_fold_audit(
    args: argparse.Namespace,
    plan: FoldPlan,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray] | None,
    test: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "train_subject_ids": plan.train_subject_ids,
        "train_subject_raw_ids": plan.train_subject_raw_ids,
        "val_subject_ids": plan.val_subject_ids,
        "val_subject_raw_ids": plan.val_subject_raw_ids,
        "test_subject_ids": plan.test_subject_ids,
        "test_subject_raw_ids": plan.test_subject_raw_ids,
        "validation_mode": args.validation_mode,
        "checkpoint_policy": args.checkpoint_policy,
        "counts": {
            "train": plan.train_counts,
            "val": plan.val_counts,
            "test": plan.test_counts,
        },
        "pre_preprocess_nan_inf": audit_nan_inf(train=train, val=val, test=test),
        "target_labels_for_model_selection": False,
        "adaptation_protocol": args.adaptation_protocol,
        "target_unlabeled_used_for_adaptation": False,
        "target_bn_stats_recomputed_on_target": False,
        "target_labels_used_for_adaptation": False,
        "target_labels_used_for_model_selection": False,
        "target_labels_used_for_evaluation_only": True,
        "adaptation_eval_split": "n/a",
    }


def audit_nan_inf(**named_arrays: dict[str, np.ndarray] | None) -> dict[str, dict[str, int]]:
    out = {}
    for name, arrays in named_arrays.items():
        if arrays is None:
            continue
        out[name] = {
            "nan": int(np.isnan(arrays["x"]).sum()),
            "inf": int(np.isinf(arrays["x"]).sum()),
        }
    return out


def save_epoch_metrics_report(plan: FoldPlan, rows: list[dict[str, object]]) -> None:
    path = epoch_metrics_report_path(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def fold_artifacts(plan: FoldPlan) -> dict[str, str]:
    artifacts = {
        "prediction_path": str(plan.prediction_path),
        "standard_prediction_path": str(report_prediction_path(plan)),
        "checkpoint_path": str(plan.checkpoint_path),
        "fold_dir": str(fold_artifact_dir(plan)),
        "source_checkpoint_path": str(fold_source_checkpoint_path(plan)),
        "normalization_stats_path": str(normalization_stats_path(plan)),
        "split_info_path": str(split_info_path(plan)),
        "class_weights_path": str(class_weights_path(plan)),
        "train_log_path": str(train_log_path(plan)),
        "summary_path": str(plan.summary_path),
        "manifest_path": str(plan.manifest_path),
        "source_checkpoint_manifest_path": str(source_checkpoint_manifest_path(plan)),
        "adaptation_manifest_path": str(adaptation_manifest_path(plan)),
        "epoch_metrics_path": str(epoch_metrics_report_path(plan)),
        "fold_audit_path": str(split_audit_report_path(plan)),
    }
    adabn_ckpt = adapted_checkpoint_path(plan, "adabn")
    adabn_report = adabn_report_path(plan)
    if adabn_ckpt.exists():
        artifacts["adapted_checkpoint_path"] = str(adabn_ckpt)
    if adabn_report.exists():
        artifacts["adabn_report_path"] = str(adabn_report)
    return artifacts


def save_fold_audit(plan: FoldPlan, payload: dict[str, object]) -> None:
    write_json(split_audit_report_path(plan), payload)


def save_fold_reporting(
    args: argparse.Namespace,
    plan: FoldPlan,
    metrics: dict[str, object],
    selected_epoch: int,
    selected_reason: str,
) -> None:
    path = fold_metrics_report_path(plan)
    row = {
        "run_id": plan.run_id,
        "dataset": plan.dataset,
        "model": plan.model_name,
        "method": args.method,
        "label_protocol": plan.label_protocol,
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "selected_epoch": selected_epoch,
        "selected_reason": selected_reason,
        "validation_mode": args.validation_mode,
        "checkpoint_policy": args.checkpoint_policy,
        "test_samples": plan.test_counts["usable"],
        "alert_count": plan.test_counts["alert"],
        "fatigue_count": plan.test_counts["fatigue"],
        **{key: metrics.get(key) for key in DISPLAY_METRIC_KEYS},
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tp": metrics.get("tp"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        df = pd.read_csv(path)
        df = df.loc[df["target_subject"].astype(str) != str(plan.target_subject)]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.sort_values("target_subject", inplace=True)
    df.to_csv(path, index=False)


def update_aggregate_metrics_report(plan: FoldPlan) -> None:
    fold_path = fold_metrics_report_path(plan)
    if not fold_path.exists():
        return
    df = pd.read_csv(fold_path)
    metrics = {}
    for key in CORE_METRIC_KEYS:
        values = pd.to_numeric(df[key], errors="coerce").dropna()
        if len(values) == 0:
            continue
        metrics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    payload = {
        "completed_folds": int(len(df)),
        "metrics": metrics,
    }
    if "macro_f1" in df.columns and len(df):
        best = df.loc[pd.to_numeric(df["macro_f1"], errors="coerce").idxmax()]
        worst = df.loc[pd.to_numeric(df["macro_f1"], errors="coerce").idxmin()]
        payload["best_fold"] = {
            "target_subject": int(best["target_subject"]),
            "target_subject_raw": best.get("target_subject_raw"),
            "macro_f1": float(best["macro_f1"]),
        }
        payload["worst_fold"] = {
            "target_subject": int(worst["target_subject"]),
            "target_subject_raw": worst.get("target_subject_raw"),
            "macro_f1": float(worst["macro_f1"]),
        }
    write_json(aggregate_metrics_report_path(plan), payload)


def update_artifacts_report(plan: FoldPlan) -> None:
    root = run_dir(plan)
    write_json(
        root / "artifacts.json",
        {
            "run_dir": root,
            "run_config": root / "run_config.json",
            "reproducibility": root / "reproducibility.json",
            "model_selection_policy": root / "model_selection_policy.json",
            "split_audit_dir": root / "split_audit",
            "metrics_dir": root / "metrics",
            "predictions_dir": root / "predictions",
            "checkpoints_dir": root / "checkpoints",
            "summaries_dir": root / "summaries",
            "reports_dir": root / "reports",
            "source_checkpoint_manifest": root / "checkpoint_manifest.csv",
            "adaptation_manifest": root / "adaptation_manifest.csv",
        },
    )


def print_fold_result(args: argparse.Namespace, metrics: dict[str, object], plan: FoldPlan) -> None:
    if not should_log(args, "normal"):
        return
    print("")
    print("Fold result")
    print(format_metrics_inline(metrics, ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "auprc"]))
    print(format_metrics_inline(metrics, ["fatigue_precision", "fatigue_recall", "specificity"]))
    print("confusion matrix (rows=true, cols=pred; labels=[alert, fatigue]):")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    print(f"[[{matrix[0, 0]}, {matrix[0, 1]}],")
    print(f" [{matrix[1, 0]}, {matrix[1, 1]}]]")
    print("")
    print("Evaluation label audit:")
    print(f"alert={plan.test_counts['alert']} | fatigue={plan.test_counts['fatigue']}")
    print("")


def format_metrics_inline(metrics: dict[str, object], keys: list[str]) -> str:
    labels = {
        "balanced_accuracy": "balanced_acc",
        "roc_auc": "auc",
    }
    return " | ".join(f"{labels.get(key, key)}={format_metric(metrics.get(key, np.nan))}" for key in keys)


def print_final_aggregate_summary(args: argparse.Namespace, plans: list[FoldPlan]) -> None:
    if not plans or not should_log(args, "quiet"):
        return
    path = aggregate_metrics_report_path(plans[0])
    root = run_dir(plans[0])
    if not path.exists():
        if should_log(args, "normal"):
            print(f"Artifacts saved to: {root}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    run_config_path = root / "run_config.json"
    total_subjects = len(plans)
    if run_config_path.exists():
        try:
            total_subjects = len(json.loads(run_config_path.read_text(encoding="utf-8")).get("subjects", [])) or total_subjects
        except json.JSONDecodeError:
            total_subjects = len(plans)
    fold_path = fold_metrics_report_path(plans[0])
    fold_df = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    print("Final Selected-LOSO Summary")
    print("-" * 27)
    print(f"Completed selected folds: {payload.get('completed_folds', 0)} / {len(plans)}")
    print(f"LOSO coverage           : {payload.get('completed_folds', 0)} / {total_subjects} subjects")
    print(f"Selected targets        : {', '.join(f'{plan.target_subject}(raw={plan.target_subject_raw})' for plan in plans)}")
    if not fold_df.empty and should_log(args, "normal"):
        print("")
        print("Per-fold results")
        table_columns = [
            "target_subject",
            "target_subject_raw",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "fatigue_recall",
            "specificity",
            "roc_auc",
            "auprc",
        ]
        table = fold_df[[column for column in table_columns if column in fold_df.columns]].copy()
        table.rename(
            columns={
                "target_subject": "subject",
                "target_subject_raw": "raw_id",
                "balanced_accuracy": "balanced_acc",
            },
            inplace=True,
        )
        for column in table.columns:
            if column not in {"subject", "raw_id"}:
                table[column] = pd.to_numeric(table[column], errors="coerce").map(lambda value: f"{value:.4f}")
        print(table.to_string(index=False))
    print("")
    print("Aggregate results")
    print("metric             mean     std      min      max")
    for key in CORE_METRIC_KEYS:
        item = metrics.get(key)
        if not item:
            continue
        print(f"{key:<18} {item['mean']:.4f}   {item['std']:.4f}   {item['min']:.4f}   {item['max']:.4f}")
    if "best_fold" in payload and should_log(args, "normal"):
        best = payload["best_fold"]
        worst = payload["worst_fold"]
        print("")
        print(f"Best fold  : subject {best['target_subject']} (raw={best['target_subject_raw']}) | macro_f1={best['macro_f1']:.4f}")
        print(f"Worst fold : subject {worst['target_subject']} (raw={worst['target_subject_raw']}) | macro_f1={worst['macro_f1']:.4f}")
    print("")
    print("Artifacts saved to:")
    print(root)


def save_success_summary_row(
    path: Path,
    *,
    args: argparse.Namespace,
    plan: FoldPlan,
    metrics: dict[str, object],
    class_weights: np.ndarray | None,
    best_epoch: int | None,
    best_macro_f1: float,
    best_balanced_acc: float,
    best_monitor: float,
    selected_epoch: int,
    selected_reason: str,
) -> None:
    row = {
        **base_summary_fields(args, plan),
        "status": "success",
        "error": "",
        "train_count": plan.train_counts["usable"],
        "val_count": plan.val_counts["usable"],
        "test_count": plan.test_counts["usable"],
        "test_samples": plan.test_counts["usable"],
        "alert_count": plan.test_counts["alert"],
        "fatigue_count": plan.test_counts["fatigue"],
        "train_alert_count": plan.train_counts["alert"],
        "train_fatigue_count": plan.train_counts["fatigue"],
        "val_alert_count": plan.val_counts["alert"],
        "val_fatigue_count": plan.val_counts["fatigue"],
        "test_alert_count": plan.test_counts["alert"],
        "test_fatigue_count": plan.test_counts["fatigue"],
        "class_weight_0": None if class_weights is None else float(class_weights[0]),
        "class_weight_1": None if class_weights is None else float(class_weights[1]),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_balanced_accuracy": best_balanced_acc,
        "best_val_metric": best_monitor,
        "selected_epoch": selected_epoch,
        "selected_reason": selected_reason,
        "monitor_metric": args.monitor_metric,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "fatigue_precision": metrics["fatigue_precision"],
        "fatigue_recall": metrics["fatigue_recall"],
        "fatigue_f1": metrics["fatigue_f1"],
        "alert_precision": metrics["alert_precision"],
        "alert_recall": metrics["alert_recall"],
        "alert_f1": metrics["alert_f1"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "miss_rate": metrics["miss_rate"],
        "roc_auc": metrics["roc_auc"],
        "auprc": metrics["auprc"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "majority_class": metrics["majority_class"],
        "majority_accuracy": metrics["majority_accuracy"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    upsert_summary_row(path, row)
    write_overall_metrics(path, plan.dataset, plan.model_name, plan.label_protocol, args.method)


def write_failed_summary_row(path: Path, args: argparse.Namespace, plan: FoldPlan, exc: Exception) -> None:
    row = {
        **base_summary_fields(args, plan),
        "status": "failed",
        "error": repr(exc),
        "train_count": plan.train_counts["usable"],
        "val_count": plan.val_counts["usable"],
        "test_count": plan.test_counts["usable"],
        "train_alert_count": plan.train_counts["alert"],
        "train_fatigue_count": plan.train_counts["fatigue"],
        "val_alert_count": plan.val_counts["alert"],
        "val_fatigue_count": plan.val_counts["fatigue"],
        "test_alert_count": plan.test_counts["alert"],
        "test_fatigue_count": plan.test_counts["fatigue"],
    }
    upsert_summary_row(path, row)


def write_checkpoint_manifest_row(
    path: Path,
    args: argparse.Namespace,
    plan: FoldPlan,
    *,
    status: str,
    best_epoch: int | None,
    best_val_metric: float | None,
    error: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "dataset": plan.dataset,
        "model": plan.model_name,
        "method": args.method,
        "adaptation_protocol": args.adaptation_protocol,
        "input_channels": plan.input_channels,
        "input_samples": plan.input_samples,
        "num_classes": plan.num_classes,
        "label_protocol": plan.label_protocol,
        "label_mode": args.label_mode,
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "seed": args.seed,
        "class_balance": args.class_balance,
        "loss_type": args.loss_type,
        "epochs": args.epochs,
        "deterministic": args.deterministic,
        "num_workers": args.num_workers,
        "validation_mode": args.validation_mode,
        "val_ratio": args.val_ratio,
        "val_subject_ratio": args.val_subject_ratio,
        "checkpoint_policy": args.checkpoint_policy,
        "fixed_eval_epoch": args.fixed_eval_epoch,
        "disable_early_stop": args.disable_early_stop,
        "early_stop_enabled": early_stop_enabled(args),
        "test_every_epochs": args.test_every_epochs,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "monitor_metric": args.monitor_metric,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "train_sample_count": plan.train_counts["usable"],
        "val_sample_count": plan.val_counts["usable"],
        "train_subject_count": len(plan.train_subject_ids),
        "val_subject_count": len(plan.val_subject_ids),
        "eegnet_f1": args.eegnet_f1,
        "eegnet_d": args.eegnet_d,
        "eegnet_f2": args.eegnet_f2,
        "eegnet_temporal_kernel": args.eegnet_temporal_kernel,
        "eegnet_separable_kernel": args.eegnet_separable_kernel,
        "eegnet_pool1": args.eegnet_pool1,
        "eegnet_pool2": args.eegnet_pool2,
        "eegnet_dropout": args.eegnet_dropout,
        "eegnet_norm_rate": args.eegnet_norm_rate,
        "checkpoint_path": str(plan.checkpoint_path),
        "fold_dir": str(fold_artifact_dir(plan)),
        "source_checkpoint_path": str(fold_source_checkpoint_path(plan)),
        "normalization_stats_path": str(normalization_stats_path(plan)),
        "split_info_path": str(split_info_path(plan)),
        "class_weights_path": str(class_weights_path(plan)),
        "adapted_checkpoint_path": str(adapted_checkpoint_path(plan, "adabn")) if args.method == "adabn" else "",
        "prediction_csv_path": str(plan.prediction_path),
        "summary_path": str(plan.summary_path),
        "command": plan.command,
        "status": status,
        "error": error,
    }
    new_row = pd.DataFrame([row])
    if path.exists():
        manifest = pd.read_csv(path)
        if manifest_has_run_id(path, plan.run_id):
            if not args.overwrite:
                raise FileExistsError(f"Run ID already exists in manifest: {plan.run_id}")
            manifest = manifest.loc[manifest["run_id"].astype(str) != str(plan.run_id)]
        for col in new_row.columns:
            if col not in manifest.columns:
                manifest[col] = np.nan
        for col in manifest.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        manifest = pd.concat([manifest, new_row[manifest.columns]], ignore_index=True)
    else:
        manifest = new_row
    manifest.to_csv(path, index=False)


def dataset_path_for_manifest(args: argparse.Namespace) -> str:
    if args.dataset == "standard-npz":
        return "" if args.standard_npz_path is None else str(args.standard_npz_path)
    if args.dataset == "sadt":
        return str(args.sadt_path)
    if args.raw_data_dir is not None and args.label_dir is not None:
        return f"raw_data_dir={args.raw_data_dir};label_dir={args.label_dir}"
    return str(args.data_root)


def write_source_checkpoint_manifest_row(
    path: Path,
    args: argparse.Namespace,
    plan: FoldPlan,
    *,
    checkpoint_path: Path,
    best_epoch: int | None,
    best_val_metric: float | None,
    selected_epoch: int,
    selected_reason: str,
) -> None:
    row = {
        "run_id": plan.run_id,
        "dataset_path": dataset_path_for_manifest(args),
        "dataset_name": plan.dataset,
        "protocol_name": plan.label_protocol,
        "label_protocol": plan.label_protocol,
        "model_name": plan.model_name,
        "method": "source_only",
        "seed": args.seed,
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "target_id_space": args.target_id_space,
        "fold_dir": str(fold_artifact_dir(plan)),
        "checkpoint_path": str(checkpoint_path),
        "legacy_checkpoint_path": str(plan.checkpoint_path),
        "normalization_stats_path": str(normalization_stats_path(plan)),
        "split_info_path": str(split_info_path(plan)),
        "class_weights_path": str(class_weights_path(plan)),
        "fold_config_path": str(fold_config_path(plan)),
        "train_log_path": str(train_log_path(plan)),
        "checkpoint_policy": args.checkpoint_policy,
        "monitor_metric": args.monitor_metric,
        "best_epoch": best_epoch,
        "selected_epoch": selected_epoch,
        "selected_reason": selected_reason,
        "val_metric_value": best_val_metric,
        "train_subjects": json.dumps(plan.train_subject_ids),
        "train_subject_raw_ids": json.dumps(jsonable(plan.train_subject_raw_ids)),
        "val_subjects": json.dumps(plan.val_subject_ids),
        "val_subject_raw_ids": json.dumps(jsonable(plan.val_subject_raw_ids)),
        "test_subjects": json.dumps(plan.test_subject_ids),
        "test_subject_raw_ids": json.dumps(jsonable(plan.test_subject_raw_ids)),
        "created_at": plan.created_at,
    }
    upsert_manifest_row(path, row, key_cols=["run_id"])


def write_adaptation_manifest_row(
    path: Path,
    args: argparse.Namespace,
    plan: FoldPlan,
    *,
    source_checkpoint_path: Path,
    adapted_checkpoint_path_value: Path | None,
    prediction_path: Path,
    metrics_path: Path,
    adaptation_report_path: Path,
    target_samples_used: int,
) -> None:
    row = {
        "run_id": plan.run_id,
        "method": args.method,
        "protocol_name": plan.label_protocol,
        "label_protocol": plan.label_protocol,
        "model_name": plan.model_name,
        "seed": args.seed,
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "source_checkpoint_path": str(source_checkpoint_path),
        "adapted_checkpoint_path": "" if adapted_checkpoint_path_value is None else str(adapted_checkpoint_path_value),
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "adaptation_report_path": str(adaptation_report_path),
        "target_samples_used": target_samples_used,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    upsert_manifest_row(path, row, key_cols=["method", "seed", "target_subject"])


def upsert_manifest_row(path: Path, row: dict[str, object], *, key_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        manifest = pd.read_csv(path)
        for col in key_cols:
            if col not in manifest.columns:
                manifest[col] = np.nan
        mask = pd.Series([True] * len(manifest))
        for col in key_cols:
            mask &= manifest[col].astype(str) == str(row[col])
        manifest = manifest.loc[~mask]
        for col in new_row.columns:
            if col not in manifest.columns:
                manifest[col] = np.nan
        for col in manifest.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        manifest = pd.concat([manifest, new_row[manifest.columns]], ignore_index=True)
    else:
        manifest = new_row
    manifest.to_csv(path, index=False)


def write_overall_metrics(summary_path: Path, dataset: str, model_name: str, label_protocol: str, method_name: str) -> None:
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if "status" in summary.columns:
        summary = summary[summary["status"] == "success"]
    if summary.empty:
        return

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "fatigue_recall",
        "sensitivity",
        "alert_recall",
        "specificity",
        "miss_rate",
        "roc_auc",
        "auprc",
    ]
    rows = []
    lines = [
        f"{dataset} {model_name} {method_name} overall metrics ({label_protocol})",
        "Primary aggregation: subject-wise mean +/- std across completed LOSO folds.",
        f"completed_folds={len(summary)}",
        "",
    ]
    for metric_name in metric_names:
        if metric_name not in summary.columns:
            continue
        values = pd.to_numeric(summary[metric_name], errors="coerce").to_numpy(dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            mean = np.nan
            std = np.nan
        else:
            mean = float(np.mean(valid))
            std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        rows.append({"metric": metric_name, "mean": mean, "std": std, "n": int(len(valid))})
        lines.append(f"{metric_name}: mean={mean:.6f} std={std:.6f} n={len(valid)}")

    stem = f"{dataset}_{model_name}_{method_name}_{label_protocol}_overall_metrics"
    txt_path = summary_path.parent / f"{stem}.txt"
    csv_path = summary_path.parent / f"{stem}.csv"
    txt_path.write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def base_summary_fields(args: argparse.Namespace, plan: FoldPlan) -> dict[str, object]:
    return {
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "dataset": plan.dataset,
        "model": plan.model_name,
        "method": args.method,
        "adaptation_protocol": args.adaptation_protocol,
        "input_channels": plan.input_channels,
        "input_samples": plan.input_samples,
        "num_classes": plan.num_classes,
        "label_protocol": plan.label_protocol,
        "target_subject": plan.target_subject,
        "target_subject_raw": plan.target_subject_raw,
        "label_mode": args.label_mode,
        "seed": args.seed,
        "class_balance": args.class_balance,
        "loss_type": args.loss_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "bandpass": args.bandpass,
        "robust_clip": args.robust_clip,
        "deterministic": args.deterministic,
        "debug_repro": args.debug_repro,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_scheduler": args.lr_scheduler,
        "early_stop_patience": args.early_stop_patience,
        "validation_mode": args.validation_mode,
        "val_ratio": args.val_ratio,
        "val_subject_ratio": args.val_subject_ratio,
        "train_subject_count": len(plan.train_subject_ids),
        "val_subject_count": len(plan.val_subject_ids),
        "train_subject_ids": plan.train_subject_ids,
        "train_subject_raw_ids": plan.train_subject_raw_ids,
        "val_subject_ids": plan.val_subject_ids,
        "val_subject_raw_ids": plan.val_subject_raw_ids,
        "train_sample_count": plan.train_counts["usable"],
        "val_sample_count": plan.val_counts["usable"],
        "checkpoint_policy": args.checkpoint_policy,
        "fixed_eval_epoch": args.fixed_eval_epoch,
        "disable_early_stop": args.disable_early_stop,
        "early_stop_enabled": early_stop_enabled(args),
        "test_every_epochs": args.test_every_epochs,
        "min_delta": args.min_delta,
        "monitor_metric": args.monitor_metric,
        "eegnet_f1": args.eegnet_f1,
        "eegnet_d": args.eegnet_d,
        "eegnet_f2": args.eegnet_f2,
        "eegnet_temporal_kernel": args.eegnet_temporal_kernel,
        "eegnet_separable_kernel": args.eegnet_separable_kernel,
        "eegnet_pool1": args.eegnet_pool1,
        "eegnet_pool2": args.eegnet_pool2,
        "eegnet_dropout": args.eegnet_dropout,
        "eegnet_norm_rate": args.eegnet_norm_rate,
        "n_train_subjects": len(plan.train_subject_ids),
        "n_val_subjects": len(plan.val_subject_ids),
    }


def upsert_summary_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_cols = [
        "dataset",
        "model",
        "method",
        "label_protocol",
        "label_mode",
        "target_subject",
        "seed",
        "class_balance",
        "optimizer",
        "lr",
        "weight_decay",
        "monitor_metric",
        "validation_mode",
        "checkpoint_policy",
        "eegnet_f1",
        "eegnet_d",
        "eegnet_f2",
        "eegnet_temporal_kernel",
        "eegnet_separable_kernel",
        "eegnet_pool1",
        "eegnet_pool2",
        "eegnet_dropout",
        "eegnet_norm_rate",
    ]
    new_row = pd.DataFrame([row])
    if path.exists():
        summary = pd.read_csv(path)
        for col in new_row.columns:
            if col not in summary.columns:
                summary[col] = row[col] if col in key_cols else np.nan
        for col in key_cols:
            if col not in summary.columns:
                summary[col] = row.get(col, "")
        for col in summary.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        mask = np.ones(len(summary), dtype=bool)
        for col in key_cols:
            mask &= summary[col].astype(str) == str(row[col])
        summary = summary.loc[~mask]
        summary = pd.concat([summary, new_row[summary.columns]], ignore_index=True)
    else:
        summary = new_row
    summary.sort_values(["dataset", "label_protocol", "target_subject", "seed", "class_balance"], inplace=True)
    summary.to_csv(path, index=False)


def print_global_plan_header(args: argparse.Namespace, context: DatasetContext, target_subjects: list[int]) -> None:
    print("global_loso_plan")
    print(f"  dataset={context.dataset}")
    print(f"  model={context.model_name}")
    print(f"  input_channels={context.input_channels}")
    print(f"  input_samples={context.input_samples}")
    print(f"  num_classes={context.num_classes}")
    print(f"  label_protocol={context.label_protocol}")
    print(f"  label_mode={args.label_mode}")
    if context.dataset != "seedvig":
        print("  label_mode_applicable=False")
    print(f"  class_balance={args.class_balance}")
    print(f"  loss_type={args.loss_type}")
    print(f"  optimizer={args.optimizer}")
    print(f"  lr={args.lr}")
    print(f"  weight_decay={args.weight_decay}")
    print(f"  early_stop_patience={args.early_stop_patience}")
    print(f"  monitor_metric={args.monitor_metric}")
    print(f"  validation_mode={args.validation_mode}")
    print(f"  val_subject_ratio={args.val_subject_ratio}")
    print(f"  val_ratio={args.val_ratio}")
    print(f"  checkpoint_policy={args.checkpoint_policy}")
    print(f"  fixed_eval_epoch={args.fixed_eval_epoch}")
    print(f"  test_every_epochs={args.test_every_epochs}")
    print(f"  early_stop_enabled={early_stop_enabled(args)}")
    print(
        "  eegnet="
        f"f1:{args.eegnet_f1},d:{args.eegnet_d},f2:{args.eegnet_f2},"
        f"temporal_kernel:{args.eegnet_temporal_kernel},"
        f"separable_kernel:{args.eegnet_separable_kernel},"
        f"pool1:{args.eegnet_pool1},pool2:{args.eegnet_pool2},"
        f"dropout:{args.eegnet_dropout},norm_rate:{args.eegnet_norm_rate}"
    )
    print(f"  data_root={args.data_root}")
    print(f"  raw_data_dir={args.raw_data_dir}")
    print(f"  label_dir={args.label_dir}")
    print(f"  output_dir={args.output_dir}")
    print(f"  outputs_enabled={outputs_enabled(args)}")
    print(f"  run_all_loso={args.run_all_loso}")
    print(f"  max_folds={args.max_folds}")
    print(f"  dry_run={args.dry_run}")
    print(f"  selected_targets={target_subjects}")
    print(f"  selected_target_raw_ids={raw_subject_ids(context, target_subjects)}")
    if context.subject_mapping:
        print(f"  subject_mapping={context.subject_mapping}")
    print(f"  included_subject_count={len(context.subjects)}")
    if context.integrity_report is not None:
        print(f"  included_session_count={len(context.integrity_report.valid_file_pairs)}")
        print(f"  label_rule={context.integrity_report.label_rule}")
    elif context.label_protocol == "rt_binary":
        print(f"  included_session_count={len(context.subjects)}")
        print("  label_rule=rt_binary: 0 alert, 1 fatigue/drowsy")
    else:
        print(f"  included_session_count={len(context.subjects)}")
        print("  label_rule=standard: integer class IDs, binary source-only metrics expect 0 alert and 1 fatigue")


def print_model_selection_policy(args: argparse.Namespace) -> None:
    print("model_selection_policy")
    print(f"  validation_mode={args.validation_mode}")
    print(f"  checkpoint_policy={args.checkpoint_policy}")
    print(f"  early_stop_enabled={early_stop_enabled(args)}")
    if args.validation_mode == "none":
        print("  validation_metrics=disabled")
        print("  model_selection_source=no validation set")
        print("  target_labels_for_model_selection=False")
        if args.test_every_epochs > 0:
            print(
                "  target_interval_diagnostics=audit_only; best target diagnostic epoch is reported for analysis "
                "but is never selected automatically"
            )
        if args.checkpoint_policy == "last" and args.epochs >= 100:
            print(
                "  warning=with validation_mode=none and checkpoint_policy=last, long runs may overtrain; "
                "use fixed_epoch for a pre-declared epoch or add source-only validation for checkpoint selection"
            )
    else:
        print(f"  model_selection_source=source validation data via {args.validation_mode}")
        print("  target_labels_for_model_selection=False")


def print_fold_plan(plan: FoldPlan, *, dry_run: bool) -> None:
    prefix = "dry_run_fold_plan" if dry_run else "fold_plan"
    print(prefix)
    print(f"  dataset={plan.dataset}")
    print(f"  model={plan.model_name}")
    print(f"  input_channels={plan.input_channels}")
    print(f"  input_samples={plan.input_samples}")
    print(f"  num_classes={plan.num_classes}")
    print(f"  label_protocol={plan.label_protocol}")
    print(f"  target_subject={plan.target_subject}")
    print(f"  target_subject_raw={plan.target_subject_raw}")
    print(f"  validation_mode={plan.validation_mode}")
    print(f"  validation_strategy={plan.validation_strategy}")
    print(f"  checkpoint_policy={plan.checkpoint_policy}")
    print(f"  train_subject_ids={plan.train_subject_ids}")
    print(f"  train_subject_raw_ids={plan.train_subject_raw_ids}")
    print(f"  val_subject_ids={plan.val_subject_ids}")
    print(f"  val_subject_raw_ids={plan.val_subject_raw_ids}")
    print(f"  test_subject_ids={plan.test_subject_ids}")
    print(f"  test_subject_raw_ids={plan.test_subject_raw_ids}")
    for name, counts in (("train", plan.train_counts), ("val", plan.val_counts), ("test_audit_only", plan.test_counts)):
        print(
            f"  {name}: sessions={counts['sessions']} usable={counts['usable']} "
            f"alert={counts['alert']} fatigue={counts['fatigue']} excluded={counts['excluded']}"
        )
    if plan.outputs_enabled:
        print(f"  predictions={plan.prediction_path}")
        print(f"  checkpoint={plan.checkpoint_path}")
    else:
        print("  predictions=disabled")
        print("  checkpoint=disabled")
    if plan.latest_checkpoint_path is not None and plan.outputs_enabled:
        print(f"  latest_checkpoint={plan.latest_checkpoint_path}")
    if plan.outputs_enabled:
        print(f"  summary={plan.summary_path}")
        print(f"  fold_report={plan.fold_report_path}")
        print(f"  val_metrics={plan.val_metrics_path}")
        print(f"  test_metrics_history={plan.test_metrics_path}")
        print(f"  manifest={plan.manifest_path}")
    else:
        print("  summary=disabled")
        print("  fold_report=disabled")
        print("  val_metrics=disabled")
        print("  test_metrics_history=disabled")
        print("  manifest=disabled")
    print(f"  run_id={plan.run_id}")
    print(f"  single_fold_command={plan.single_fold_command}")
    if dry_run:
        print("  target_counts_are_audit_only=True")
        print("  loads_full_eeg_tensors=False")
        print("  instantiates_model=False")


def print_discovery_sanity(file_pairs, subjects: list[int]) -> None:
    print(f"subjects={len(subjects)} ids={subjects}")
    print(f"sessions={len(file_pairs)}")
    print("expected_segment_shape=(n_segments, 17, 1600)")
    print("confirmed_each_seedvig_segment_shape=(17, 1600)")
    print("sample_rate=200")
    print("additional_downsampling_applied=False")


def print_source_sanity(train_sessions, val_sessions, context: DatasetContext) -> None:
    if context.dataset != "seedvig":
        print(f"final_segment_shape_per_sample=({context.input_channels}, {context.input_samples})")
        print("source_raw_nan_count=0 source_raw_inf_count=0")
        return
    sessions = list(train_sessions) + list(val_sessions)
    label_values = np.concatenate([s.y for s in sessions])
    raw_segment_counts = [s.raw_segment_count for s in sessions]
    nan_total = sum(s.nan_count for s in sessions)
    inf_total = sum(s.inf_count for s in sessions)
    first_shape = sessions[0].x.shape if sessions else None
    values, counts = np.unique(label_values, return_counts=True)
    distribution = dict(zip(values.tolist(), counts.tolist(), strict=False))
    print(f"first_source_session_segment_shape={first_shape}")
    print("final_segment_shape_per_sample=(17, 1600)")
    print(f"raw_segments_per_session={sorted(set(raw_segment_counts))}")
    print(f"source_label_distribution_after_threshold={distribution}")
    print(f"source_excluded_sample_count={sum(s.dropped_middle_count for s in sessions)}")
    print(f"source_raw_nan_count={nan_total} source_raw_inf_count={inf_total}")


def print_source_split_sanity(train, val) -> None:
    val_count = 0 if val is None else len(val["y"])
    val_subjects = [] if val is None else sorted(int(s) for s in set(val["subject_id"]))
    print(f"split_counts train={len(train['y'])} val={val_count}")
    print(
        "split_subjects "
        f"train={sorted(int(s) for s in set(train['subject_id']))} "
        f"val={val_subjects}"
    )
    named_arrays = [("train", train)]
    if val is not None:
        named_arrays.append(("val", val))
    for name, arrays in named_arrays:
        values, counts = np.unique(arrays["y"], return_counts=True)
        print(f"{name}_label_distribution={dict(zip(values.tolist(), counts.tolist(), strict=False))}")


def print_target_sanity(test_sessions, test, context: DatasetContext) -> None:
    if context.dataset != "seedvig":
        values, counts = np.unique(test["y"], return_counts=True)
        distribution = dict(zip(values.tolist(), counts.tolist(), strict=False))
        print(f"target_count={len(test['y'])}")
        print(f"confirmed_target_segment_shape_per_sample=({context.input_channels}, {context.input_samples})")
        print(f"target_label_distribution={distribution}")
        print("target_raw_nan_count=0 target_raw_inf_count=0")
        return
    raw_segment_counts = [s.raw_segment_count for s in test_sessions]
    nan_total = sum(s.nan_count for s in test_sessions)
    inf_total = sum(s.inf_count for s in test_sessions)
    first_shape = test_sessions[0].x.shape if test_sessions else None
    values, counts = np.unique(test["y"], return_counts=True)
    distribution = dict(zip(values.tolist(), counts.tolist(), strict=False))
    print(f"target_session_segment_shape={first_shape}")
    print("confirmed_target_segment_shape_per_sample=(17, 1600)")
    print(f"target_raw_segments_per_session={sorted(set(raw_segment_counts))}")
    print(f"target_count={len(test['y'])}")
    print(f"target_label_distribution_after_threshold={distribution}")
    print(f"target_excluded_sample_count={sum(s.dropped_middle_count for s in test_sessions)}")
    print(f"target_raw_nan_count={nan_total} target_raw_inf_count={inf_total}")


def print_nan_inf_after_preprocessing(*named_arrays) -> None:
    for name, arrays in named_arrays:
        nan_count, inf_count = nan_inf_counts(arrays["x"])
        print(f"{name}_post_preprocess_nan={nan_count} inf={inf_count}")


def print_final_metrics(metrics: dict[str, object]) -> None:
    print("target_metrics")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "fatigue_precision",
        "fatigue_recall",
        "sensitivity",
        "alert_recall",
        "specificity",
        "miss_rate",
        "roc_auc",
        "auprc",
    ):
        value = float(metrics[key])
        print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: nan")
    print("confusion_matrix")
    print(metrics["confusion_matrix"])


def print_best_target_diagnostic_metrics(epoch: int, metrics: dict[str, object]) -> None:
    print("best_target_diagnostic_auc_epoch_metrics")
    print("  selection_scope: interval target diagnostic evaluations only")
    print("  note: target diagnostics are not used for training, checkpoint selection, early stopping, or model selection")
    print(f"  epoch: {epoch}")
    for key in (
        "roc_auc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "fatigue_recall",
        "sensitivity",
        "alert_recall",
        "specificity",
        "miss_rate",
    ):
        print(f"  {key}: {format_metric(metrics[key])}")
    print("confusion_matrix")
    print(metrics["confusion_matrix"])


def print_epoch_test_metrics(target_subject: int, epoch: int, metrics: dict[str, object]) -> None:
    print(
        f"target_diagnostic_metrics target_subject={target_subject} epoch={epoch:03d} "
        f"accuracy={format_metric(metrics['accuracy'])} "
        f"balanced_accuracy={format_metric(metrics['balanced_accuracy'])} "
        f"macro_f1={format_metric(metrics['macro_f1'])} "
        f"roc_auc={format_metric(metrics['roc_auc'])} "
        f"auprc={format_metric(metrics['auprc'])} "
        f"sensitivity={format_metric(metrics['sensitivity'])} "
        f"specificity={format_metric(metrics['specificity'])} "
        f"miss_rate={format_metric(metrics['miss_rate'])}"
    )


def format_metric(value: object) -> str:
    value = float(value)
    return "nan" if np.isnan(value) else f"{value:.4f}"


def pair_subjects(pairs: list[tuple[Path, Path]]) -> list[int]:
    return sorted({parse_subject_id(raw_path) for raw_path, _ in pairs})


def counts_for_pairs(pairs: list[tuple[Path, Path]], report: IntegrityReport) -> dict[str, int]:
    session_by_id = {session.session_id: session for session in report.sessions}
    counts = {"sessions": 0, "usable": 0, "alert": 0, "fatigue": 0, "excluded": 0}
    for raw_path, _ in pairs:
        session = session_by_id[raw_path.stem]
        counts["sessions"] += 1
        counts["usable"] += session.usable_binary_samples
        counts["alert"] += session.alert_count
        counts["fatigue"] += session.fatigue_count
        counts["excluded"] += session.excluded_count
    return counts


def zero_counts() -> dict[str, int]:
    return {"sessions": 0, "usable": 0, "alert": 0, "fatigue": 0, "excluded": 0}


def stratified_metadata_counts(source_counts: dict[str, int], val_ratio: float) -> tuple[dict[str, int], dict[str, int]]:
    val_alert = int(round(source_counts["alert"] * val_ratio))
    val_fatigue = int(round(source_counts["fatigue"] * val_ratio))
    val_counts = {
        "sessions": source_counts["sessions"],
        "usable": val_alert + val_fatigue,
        "alert": val_alert,
        "fatigue": val_fatigue,
        "excluded": 0,
    }
    train_alert = source_counts["alert"] - val_alert
    train_fatigue = source_counts["fatigue"] - val_fatigue
    train_counts = {
        "sessions": source_counts["sessions"],
        "usable": train_alert + train_fatigue,
        "alert": train_alert,
        "fatigue": train_fatigue,
        "excluded": source_counts["excluded"],
    }
    return train_counts, val_counts


def print_recommended_gpu_command(args: argparse.Namespace) -> None:
    print("recommended_later_gpu_command")
    if args.dataset == "seedvig":
        path_args = ""
        if args.raw_data_dir is not None and args.label_dir is not None:
            path_args = f" --raw-data-dir {args.raw_data_dir} --label-dir {args.label_dir}"
        print(
            "python -m eegda.train --dataset seedvig --model eegnet --method source_only --protocol loso "
            "--run-all-loso --epochs 100 --batch-size 64 --device cuda --label-mode threshold35 "
            "--class-balance weighted_loss --optimizer adamw --weight-decay 0.0001 "
            "--early-stop-patience 15 --monitor-metric macro_f1"
            f"{path_args}"
        )
        return
    if args.dataset == "sadt":
        print(
            "python -m eegda.train --dataset sadt-balanced --model eegnet --method source_only --protocol loso "
            f"--sadt-balanced-path {args.sadt_path} --run-all-loso --epochs 50 --batch-size 64 "
            "--device cuda --validation-mode none --checkpoint-policy last "
            f"--class-balance {args.class_balance}"
        )
        return
    if args.dataset == "standard-npz":
        print(
            "python -m eegda.train --data my_dataset.npz "
            "--model eegnet --method source_only --protocol loso --run-all-loso --epochs 50 "
            "--batch-size 64 --device cuda --validation-mode none --checkpoint-policy last"
        )
        return


if __name__ == "__main__":
    main()
