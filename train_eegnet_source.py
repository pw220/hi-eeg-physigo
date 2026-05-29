from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
from models.eegnet import EEGNet
from utils.metrics import classification_metrics, entropy_from_probs, softmax
from utils.seed import set_seed


@dataclass(frozen=True)
class FoldPlan:
    target_subject: int
    train_pairs: list[tuple[Path, Path]]
    val_pairs: list[tuple[Path, Path]]
    test_pairs: list[tuple[Path, Path]]
    train_subject_ids: list[int]
    val_subject_ids: list[int]
    test_subject_ids: list[int]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-only EEGNet LOSO baseline on SEED-VIG raw EEG")
    parser.add_argument("--data-root", default="data/raw/SEED-VIG")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument("--run-all-loso", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--label-mode", choices=("threshold35", "strict035070"), default="threshold35")
    parser.add_argument("--class-balance", choices=("none", "weighted_loss"), default="weighted_loss")
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
    parser.add_argument("--checkpoint-policy", choices=("best_val", "last", "fixed_epoch"), default="best_val")
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
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--debug-repro", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--min-class-samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    if (args.raw_data_dir is None) != (args.label_dir is None):
        raise ValueError("--raw-data-dir and --label-dir must be provided together")
    validate_training_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    outputs_dir = Path(args.output_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Full integrity reports are saved for both label modes. Fold planning uses
    # only the selected mode report.
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
        if not args.dry_run:
            save_integrity_csv(report, outputs_dir / f"seedvig_integrity_{label_mode}.csv")

    integrity_report = reports[args.label_mode]
    file_pairs = integrity_report.valid_file_pairs
    subjects = sorted({parse_subject_id(raw_path) for raw_path, _ in file_pairs})
    target_subjects = resolve_target_subjects(args, subjects)

    print_global_plan_header(args, integrity_report, target_subjects)
    plans = [
        plan_loso_fold(
            args=args,
            integrity_report=integrity_report,
            file_pairs=file_pairs,
            target_subject=target_subject,
            outputs_dir=outputs_dir,
        )
        for target_subject in target_subjects
    ]

    if args.dry_run:
        for plan in plans:
            print_fold_plan(plan, dry_run=True)
        print_recommended_gpu_command()
        return

    device = choose_device(args.device)
    repro_metadata = reproducibility_metadata(args, device)
    print_reproducibility_metadata(repro_metadata)
    for plan in plans:
        if args.skip_existing and fold_outputs_exist(plan):
            print(f"Skipping target_subject={plan.target_subject}: existing prediction CSV and checkpoint found")
            continue
        try:
            run_loso_fold(args, integrity_report, plan, device, repro_metadata)
        except Exception as exc:  # noqa: BLE001 - all-LOSO should continue after fold failures
            print(f"Fold target_subject={plan.target_subject} failed: {exc}")
            traceback.print_exc()
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

    print_recommended_gpu_command()


def resolve_target_subjects(args: argparse.Namespace, subjects: list[int]) -> list[int]:
    if args.run_all_loso:
        selected = subjects
        if args.max_folds is not None:
            selected = selected[: args.max_folds]
        if not selected:
            raise ValueError("No target subjects available for --run-all-loso")
        return selected
    if args.target_subject not in subjects:
        raise ValueError(f"Target subject {args.target_subject} not found. Available: {subjects}")
    return [args.target_subject]


def validate_training_args(args: argparse.Namespace) -> None:
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


def make_run_id(args: argparse.Namespace, target_subject: int) -> str:
    base = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{base}_subject{target_subject}"


def fold_outputs_exist(plan: FoldPlan) -> bool:
    return plan.prediction_path.exists() and any(existing_checkpoints_for_plan(plan))


def existing_checkpoints_for_plan(plan: FoldPlan) -> list[Path]:
    pattern = (
        f"eegnet_source_only_*_subject_{plan.target_subject}_seed*.pt"
        if not plan.checkpoint_path.parent.exists()
        else f"eegnet_source_only_*_subject_{plan.target_subject}_seed*.pt"
    )
    return sorted(plan.checkpoint_path.parent.glob(pattern))


def plan_loso_fold(
    *,
    args: argparse.Namespace,
    integrity_report: IntegrityReport,
    file_pairs: list[tuple[Path, Path]],
    target_subject: int,
    outputs_dir: Path,
) -> FoldPlan:
    source_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) != target_subject]
    test_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) == target_subject]
    if args.validation_mode == "subject_split":
        train_pairs, val_pairs, test_pairs = split_loso_file_pairs(
            file_pairs,
            target_subject=target_subject,
            val_subject_ratio=args.val_subject_ratio,
            seed=args.seed,
        )
        train_counts = counts_for_pairs(train_pairs, integrity_report)
        val_counts = counts_for_pairs(val_pairs, integrity_report)
        train_subject_ids = pair_subjects(train_pairs)
        val_subject_ids = pair_subjects(val_pairs)
        validation_strategy = "deterministic source-subject split controlled by seed and val_subject_ratio"
    elif args.validation_mode == "sample_stratified":
        if not source_pairs or not test_pairs:
            raise ValueError("Invalid LOSO split produced an empty source or test partition")
        train_pairs = source_pairs
        val_pairs = []
        source_counts = counts_for_pairs(source_pairs, integrity_report)
        train_counts, val_counts = stratified_metadata_counts(source_counts, args.val_ratio)
        train_subject_ids = pair_subjects(source_pairs)
        val_subject_ids = pair_subjects(source_pairs)
        validation_strategy = "sample-level stratified validation within source subjects controlled by seed and val_ratio"
    elif args.validation_mode == "none":
        if not source_pairs or not test_pairs:
            raise ValueError("Invalid LOSO split produced an empty source or test partition")
        train_pairs = source_pairs
        val_pairs = []
        train_counts = counts_for_pairs(source_pairs, integrity_report)
        val_counts = zero_counts()
        train_subject_ids = pair_subjects(source_pairs)
        val_subject_ids = []
        validation_strategy = "no validation set; all non-target source samples used for training"
    else:
        raise ValueError(f"Unsupported validation mode: {args.validation_mode}")
    test_counts = counts_for_pairs(test_pairs, integrity_report)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = make_run_id(args, target_subject)
    checkpoints_dir = outputs_dir / "checkpoints"
    prediction_path = outputs_dir / f"eegnet_source_only_{args.label_mode}_subject_{target_subject}.csv"
    checkpoint_path = checkpoints_dir / (
        f"eegnet_source_only_{args.label_mode}_subject_{target_subject}_seed{args.seed}_{run_id}.pt"
    )
    latest_checkpoint_path = (
        checkpoints_dir / f"eegnet_source_only_{args.label_mode}_subject_{target_subject}_seed{args.seed}_latest.pt"
        if args.save_latest
        else None
    )
    summary_path = outputs_dir / f"eegnet_source_only_{args.label_mode}_summary.csv"
    fold_report_path = outputs_dir / f"loso_fold_integrity_{args.label_mode}_subject_{target_subject}.txt"
    val_metrics_path = outputs_dir / f"val_metrics_{args.label_mode}_subject_{target_subject}.csv"
    test_metrics_path = outputs_dir / f"test_metrics_history_{args.label_mode}_subject_{target_subject}.csv"
    manifest_path = outputs_dir / "checkpoints_manifest.csv"
    command = (
        "python train_eegnet_source.py "
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
        f"--device {args.device} "
        f"--label-mode {args.label_mode} "
        f"--class-balance {args.class_balance} "
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
    if args.raw_data_dir is not None and args.label_dir is not None:
        command += f" --raw-data-dir {shlex.quote(str(args.raw_data_dir))}"
        command += f" --label-dir {shlex.quote(str(args.label_dir))}"
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
        target_subject=target_subject,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=pair_subjects(test_pairs),
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
    )


def run_loso_fold(
    args: argparse.Namespace,
    integrity_report: IntegrityReport,
    plan: FoldPlan,
    device: torch.device,
    repro_metadata: dict[str, object],
) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    guard_run_outputs(args, plan)
    write_loso_fold_integrity_report(
        integrity_report,
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
    print_integrity_report_summary(
        integrity_report,
        target_subject=plan.target_subject,
        train_pairs=plan.train_pairs,
        val_pairs=plan.val_pairs,
        test_pairs=plan.test_pairs,
        train_counts=plan.train_counts,
        val_counts=plan.val_counts,
        test_counts=plan.test_counts,
    )
    print_fold_plan(plan, dry_run=False)

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
    assert plan.target_subject not in set(train["subject_id"])
    if val is not None:
        assert plan.target_subject not in set(val["subject_id"])

    print_source_sanity(train_sessions, val_sessions)
    print_source_split_sanity(train, val)
    train_x, val_x, preprocess_state = preprocess_source(
        train["x"],
        None if val is None else val["x"],
        robust_clip=args.robust_clip,
    )
    train["x"] = train_x
    if val is not None:
        val["x"] = val_x
        print_nan_inf_after_preprocessing(("train", train), ("val", val))
    else:
        print_nan_inf_after_preprocessing(("train", train))

    # Target labels are loaded after source-only preprocessing state is fixed;
    # they are used only for final evaluation and saved prediction diagnostics.
    test_sessions = load_seedvig_file_pairs(plan.test_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
    test = sessions_to_arrays(test_sessions)
    assert set(test["subject_id"]) == {plan.target_subject}
    test["x"] = preprocess_target(test["x"], preprocess_state)
    print_target_sanity(test_sessions, test)
    print_nan_inf_after_preprocessing(("test", test))

    train_loader = make_loader(
        train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = None
    if val is not None:
        val_loader = make_loader(
            val,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    test_loader = make_loader(
        test,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_metrics_history = []
    if args.test_every_epochs > 0:
        print(
            "target_interval_evaluation=diagnostic_only "
            "target labels are not used for training, checkpoint selection, early stopping, or model selection"
        )

    class_weights = compute_class_weights(train["y"], args.class_balance)
    print_class_balance(train["y"], args.class_balance, class_weights)
    criterion_weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)

    model_config = eegnet_model_config(args)
    print_model_and_training_config(args, model_config)
    set_seed(args.seed, deterministic=args.deterministic)
    model = EEGNet(**model_config).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    initial_checksum = model_parameter_checksum(model)
    if args.debug_repro:
        print(f"debug_repro initial_parameter_checksum={initial_checksum}")
        print(f"debug_repro first_20_train_sample_ids={first_shuffled_sample_ids(train, args.seed, 20)}")

    best_state = None
    best_epoch = 0
    best_monitor = -1.0
    best_tie = -1.0
    best_macro_f1 = -1.0
    best_balanced_acc = -1.0
    selected_state = None
    selected_epoch = 0
    selected_reason = ""
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )
        if args.debug_repro and epoch == 1:
            print(f"debug_repro epoch1_parameter_checksum={model_parameter_checksum(model)}")
        val_metrics = None
        monitor_value = np.nan
        if val_loader is not None:
            val_logits, val_y = predict_logits(model, val_loader, device)
            val_probs = softmax(val_logits)
            val_pred = val_logits.argmax(axis=1)
            val_metrics = classification_metrics(val_y, val_pred, val_probs[:, 1])
            monitor_value = _monitor_value(val_metrics, args.monitor_metric)
            tie_metric = "balanced_accuracy" if args.monitor_metric == "macro_f1" else "macro_f1"
            tie_value = _monitor_value(val_metrics, tie_metric)
            improved = best_state is None or monitor_value > best_monitor + args.min_delta or (
                np.isclose(monitor_value, best_monitor) and tie_value > best_tie + args.min_delta
            )
            if improved:
                best_epoch = epoch
                best_monitor = monitor_value
                best_tie = tie_value
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_balanced_acc = float(val_metrics["balanced_accuracy"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        if args.checkpoint_policy == "last":
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            selected_epoch = epoch
            selected_reason = "last"
        elif args.checkpoint_policy == "fixed_epoch" and epoch == args.fixed_eval_epoch:
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            selected_epoch = epoch
            selected_reason = f"fixed_epoch_{epoch}"

        if scheduler is not None and val_metrics is not None:
            scheduler.step(monitor_value)
        current_lr = optimizer.param_groups[0]["lr"]
        if val_metrics is None:
            print(
                f"target_subject={plan.target_subject} epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"validation_mode=none checkpoint_policy={args.checkpoint_policy} "
                f"lr={current_lr:.6g}"
            )
        else:
            print(
                f"target_subject={plan.target_subject} epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
                f"monitor={args.monitor_metric}:{monitor_value:.4f} "
                f"lr={current_lr:.6g} "
                f"no_improve={epochs_without_improvement}"
            )
        if args.test_every_epochs > 0 and epoch % args.test_every_epochs == 0:
            epoch_metrics = evaluate_current_model_on_target(model, test_loader, device)
            test_metrics_history.append(metrics_history_row(epoch, epoch_metrics, reason="interval_diagnostic"))
            print_epoch_test_metrics(plan.target_subject, epoch, epoch_metrics)
        if early_stop_enabled(args) and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"early_stopping_triggered epoch={epoch} "
                f"best_epoch={best_epoch} monitor_metric={args.monitor_metric} best_monitor={best_monitor:.4f}"
            )
            break

    if args.checkpoint_policy == "best_val":
        if best_state is None:
            raise RuntimeError("Training did not produce a best validation model")
        selected_state = best_state
        selected_epoch = best_epoch
        selected_reason = f"best_val_{args.monitor_metric}"
    if selected_state is None:
        raise RuntimeError(f"Training did not produce a checkpoint for policy={args.checkpoint_policy}")
    if args.debug_repro:
        print(f"debug_repro best_epoch={best_epoch} best_validation_metric={best_monitor:.10f}")
        print(f"debug_repro selected_epoch={selected_epoch} selected_reason={selected_reason}")
    model.load_state_dict(selected_state)
    if val is not None:
        save_validation_subject_metrics(
            model,
            val,
            plan.val_metrics_path,
            device,
            args.batch_size,
            args.num_workers,
            args.seed,
        )

    test_logits, test_y = predict_logits(model, test_loader, device)
    test_probs = softmax(test_logits)
    test_pred = test_probs.argmax(axis=1)
    test_metrics = classification_metrics(test_y, test_pred, test_probs[:, 1])
    print_final_metrics(test_metrics)
    has_validation = val_loader is not None
    if test_metrics_history:
        test_metrics_history.append(metrics_history_row(selected_epoch, test_metrics, reason="final_selected_checkpoint"))
        save_test_metrics_history(plan.test_metrics_path, test_metrics_history)

    save_predictions(plan.prediction_path, test, test_y, test_pred, test_probs)
    checkpoint = {
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "command": plan.command,
        "reproducibility": repro_metadata,
        "model_state_dict": selected_state,
        "model_config": model_config,
        "training_config": training_config(args),
        "args": vars(args),
        "label_mode": args.label_mode,
        "target_subject": plan.target_subject,
        "seed": args.seed,
        "class_balance": args.class_balance,
        "train_subject_ids": plan.train_subject_ids,
        "val_subject_ids": plan.val_subject_ids,
        "normalization_mean": torch.from_numpy(preprocess_state["mean"].copy()),
        "normalization_std": torch.from_numpy(preprocess_state["std"].copy()),
        "clipping_thresholds": tensorize_clip_bounds(preprocess_state["clip_bounds"]),
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "best_epoch": None if not has_validation else best_epoch,
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
    plan.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, plan.checkpoint_path)
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
    print(f"Saved predictions: {plan.prediction_path}")
    print(f"Saved checkpoint: {plan.checkpoint_path}")
    if plan.latest_checkpoint_path is not None:
        print(f"Saved latest checkpoint: {plan.latest_checkpoint_path}")
    print(f"Saved summary: {plan.summary_path}")
    if val is not None:
        print(f"Saved validation metrics: {plan.val_metrics_path}")
    if test_metrics_history:
        print(f"Saved target diagnostic metrics: {plan.test_metrics_path}")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
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


def preprocess_source(train_x: np.ndarray, val_x: np.ndarray | None, *, robust_clip: bool):
    clip_bounds = None
    if robust_clip:
        lo, hi = compute_robust_clip_bounds(train_x)
        train_x = apply_robust_clip(train_x, lo, hi)
        if val_x is not None:
            val_x = apply_robust_clip(val_x, lo, hi)
        clip_bounds = (lo, hi)

    mean, std = compute_channel_stats(train_x)
    state = {"clip_bounds": clip_bounds, "mean": mean, "std": std}
    train_x = apply_channel_zscore(train_x, mean, std)
    val_x = None if val_x is None else apply_channel_zscore(val_x, mean, std)
    return train_x, val_x, state


def preprocess_target(test_x: np.ndarray, state: dict[str, object]) -> np.ndarray:
    clip_bounds = state["clip_bounds"]
    if clip_bounds is not None:
        lo, hi = clip_bounds
        test_x = apply_robust_clip(test_x, lo, hi)
    return apply_channel_zscore(test_x, state["mean"], state["std"])


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


def make_seed_worker(seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def eegnet_model_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "channels": 17,
        "samples": 1600,
        "num_classes": 2,
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


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
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


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
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


def print_class_balance(y: np.ndarray, class_balance: str, class_weights: np.ndarray | None) -> None:
    counts = np.bincount(y.astype(np.int64), minlength=2)
    print("class_balance")
    print(f"  mode={class_balance}")
    print(f"  source_train_counts={{0: {int(counts[0])}, 1: {int(counts[1])}}}")
    if class_weights is None:
        print("  class_weights=None")
    else:
        print(f"  class_weights={{0: {class_weights[0]:.6f}, 1: {class_weights[1]:.6f}}}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    *,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in tqdm(loader, desc="train", leave=False):
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
    return total_loss / max(total_count, 1)


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


@torch.no_grad()
def predict_logits(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    y_list = []
    for x, y in loader:
        logits = model(x.to(device))
        logits_list.append(logits.detach().cpu().numpy())
        y_list.append(y.numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(y_list, axis=0)


def evaluate_current_model_on_target(model, loader, device: torch.device) -> dict[str, object]:
    logits, y_true = predict_logits(model, loader, device)
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    return classification_metrics(y_true, y_pred, probs[:, 1])


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
        loader = make_loader(subject_arrays, batch_size, shuffle=False, num_workers=num_workers, seed=seed)
        logits, y_true = predict_logits(model, loader, device)
        probs = softmax(logits)
        y_pred = probs.argmax(axis=1)
        metrics = classification_metrics(y_true, y_pred, probs[:, 1])
        rows.append(
            {
                "val_subject": subject_id,
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


def save_predictions(path: Path, arrays: dict[str, np.ndarray], y_true, y_pred, probs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    confidence = probs.max(axis=1)
    entropy = entropy_from_probs(probs)
    df = pd.DataFrame(
        {
            "sample_id": arrays["sample_id"],
            "subject_id": arrays["subject_id"],
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


def guard_run_outputs(args: argparse.Namespace, plan: FoldPlan) -> None:
    if plan.checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint already exists: {plan.checkpoint_path}. Use --overwrite to replace it.")
    if manifest_has_run_id(plan.manifest_path, plan.run_id) and not args.overwrite:
        raise FileExistsError(f"Run ID already exists in manifest: {plan.run_id}. Use --overwrite to replace it.")


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
    write_overall_metrics(path, args.label_mode)


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
        "label_mode": args.label_mode,
        "target_subject": plan.target_subject,
        "seed": args.seed,
        "class_balance": args.class_balance,
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


def write_overall_metrics(summary_path: Path, label_mode: str) -> None:
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
        f"EEGNet source-only overall metrics ({label_mode})",
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

    txt_path = summary_path.parent / f"eegnet_source_only_{label_mode}_overall_metrics.txt"
    csv_path = summary_path.parent / f"eegnet_source_only_{label_mode}_overall_metrics.csv"
    txt_path.write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def base_summary_fields(args: argparse.Namespace, plan: FoldPlan) -> dict[str, object]:
    return {
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "target_subject": plan.target_subject,
        "label_mode": args.label_mode,
        "seed": args.seed,
        "class_balance": args.class_balance,
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
    summary.sort_values(["label_mode", "target_subject", "seed", "class_balance"], inplace=True)
    summary.to_csv(path, index=False)


def print_global_plan_header(args: argparse.Namespace, report: IntegrityReport, target_subjects: list[int]) -> None:
    print("global_loso_plan")
    print(f"  label_mode={args.label_mode}")
    print(f"  class_balance={args.class_balance}")
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
    print(f"  run_all_loso={args.run_all_loso}")
    print(f"  max_folds={args.max_folds}")
    print(f"  dry_run={args.dry_run}")
    print(f"  selected_targets={target_subjects}")
    print(f"  included_subject_count={len(report.included_subject_ids)}")
    print(f"  included_session_count={len(report.valid_file_pairs)}")
    print(f"  label_rule={report.label_rule}")


def print_fold_plan(plan: FoldPlan, *, dry_run: bool) -> None:
    prefix = "dry_run_fold_plan" if dry_run else "fold_plan"
    print(prefix)
    print(f"  target_subject={plan.target_subject}")
    print(f"  validation_mode={plan.validation_mode}")
    print(f"  validation_strategy={plan.validation_strategy}")
    print(f"  checkpoint_policy={plan.checkpoint_policy}")
    print(f"  train_subject_ids={plan.train_subject_ids}")
    print(f"  val_subject_ids={plan.val_subject_ids}")
    print(f"  test_subject_ids={plan.test_subject_ids}")
    for name, counts in (("train", plan.train_counts), ("val", plan.val_counts), ("test_audit_only", plan.test_counts)):
        print(
            f"  {name}: sessions={counts['sessions']} usable={counts['usable']} "
            f"alert={counts['alert']} fatigue={counts['fatigue']} excluded={counts['excluded']}"
        )
    print(f"  predictions={plan.prediction_path}")
    print(f"  checkpoint={plan.checkpoint_path}")
    if plan.latest_checkpoint_path is not None:
        print(f"  latest_checkpoint={plan.latest_checkpoint_path}")
    print(f"  summary={plan.summary_path}")
    print(f"  fold_report={plan.fold_report_path}")
    print(f"  val_metrics={plan.val_metrics_path}")
    print(f"  test_metrics_history={plan.test_metrics_path}")
    print(f"  manifest={plan.manifest_path}")
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


def print_source_sanity(train_sessions, val_sessions) -> None:
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


def print_target_sanity(test_sessions, test) -> None:
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


def print_epoch_test_metrics(target_subject: int, epoch: int, metrics: dict[str, object]) -> None:
    print(
        f"target_diagnostic_metrics target_subject={target_subject} epoch={epoch:03d} "
        f"accuracy={float(metrics['accuracy']):.4f} "
        f"balanced_accuracy={float(metrics['balanced_accuracy']):.4f} "
        f"macro_f1={float(metrics['macro_f1']):.4f} "
        f"sensitivity={float(metrics['sensitivity']):.4f} "
        f"specificity={float(metrics['specificity']):.4f} "
        f"miss_rate={float(metrics['miss_rate']):.4f}"
    )


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


def print_recommended_gpu_command() -> None:
    print("recommended_later_gpu_command")
    print(
        "python train_eegnet_source.py --run-all-loso --epochs 100 --batch-size 64 "
        "--device cuda --label-mode threshold35 --class-balance weighted_loss "
        "--optimizer adamw --weight-decay 0.0001 --early-stop-patience 15 "
        "--monitor-metric macro_f1 --lr-scheduler plateau"
    )


if __name__ == "__main__":
    main()
