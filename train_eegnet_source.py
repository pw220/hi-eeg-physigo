from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    manifest_path: Path
    run_id: str
    created_at: str
    command: str
    single_fold_command: str


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
    parser.add_argument("--val-subject-ratio", type=float, default=0.2)
    parser.add_argument("--bandpass", action="store_true")
    parser.add_argument("--robust-clip", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--outputs-dir", dest="output_dir", help=argparse.SUPPRESS)
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--min-class-samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    if (args.raw_data_dir is None) != (args.label_dir is None):
        raise ValueError("--raw-data-dir and --label-dir must be provided together")
    set_seed(args.seed)
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
    for plan in plans:
        if args.skip_existing and fold_outputs_exist(plan):
            print(f"Skipping target_subject={plan.target_subject}: existing prediction CSV and checkpoint found")
            continue
        try:
            run_loso_fold(args, integrity_report, plan, device)
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
    train_pairs, val_pairs, test_pairs = split_loso_file_pairs(
        file_pairs,
        target_subject=target_subject,
        val_subject_ratio=args.val_subject_ratio,
        seed=args.seed,
    )
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
    manifest_path = outputs_dir / "checkpoints_manifest.csv"
    command = (
        "python train_eegnet_source.py "
        f"--target-subject {target_subject} "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        f"--device {args.device} "
        f"--label-mode {args.label_mode} "
        f"--class-balance {args.class_balance} "
        f"--output-dir {shlex.quote(str(args.output_dir))}"
    )
    if args.raw_data_dir is not None and args.label_dir is not None:
        command += f" --raw-data-dir {shlex.quote(str(args.raw_data_dir))}"
        command += f" --label-dir {shlex.quote(str(args.label_dir))}"
    if args.bandpass:
        command += " --bandpass"
    if args.robust_clip:
        command += " --robust-clip"

    return FoldPlan(
        target_subject=target_subject,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_subject_ids=pair_subjects(train_pairs),
        val_subject_ids=pair_subjects(val_pairs),
        test_subject_ids=pair_subjects(test_pairs),
        train_counts=counts_for_pairs(train_pairs, integrity_report),
        val_counts=counts_for_pairs(val_pairs, integrity_report),
        test_counts=counts_for_pairs(test_pairs, integrity_report),
        prediction_path=prediction_path,
        checkpoint_path=checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        summary_path=summary_path,
        fold_report_path=fold_report_path,
        manifest_path=manifest_path,
        run_id=run_id,
        created_at=created_at,
        command=" ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        single_fold_command=command,
    )


def run_loso_fold(
    args: argparse.Namespace,
    integrity_report: IntegrityReport,
    plan: FoldPlan,
    device: torch.device,
) -> None:
    set_seed(args.seed)
    guard_run_outputs(args, plan)
    write_loso_fold_integrity_report(
        integrity_report,
        plan.fold_report_path,
        target_subject=plan.target_subject,
        train_pairs=plan.train_pairs,
        val_pairs=plan.val_pairs,
        test_pairs=plan.test_pairs,
        robust_clip=args.robust_clip,
    )
    print_integrity_report_summary(
        integrity_report,
        target_subject=plan.target_subject,
        train_pairs=plan.train_pairs,
        val_pairs=plan.val_pairs,
        test_pairs=plan.test_pairs,
    )
    print_fold_plan(plan, dry_run=False)

    train_sessions = load_seedvig_file_pairs(plan.train_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
    val_sessions = load_seedvig_file_pairs(plan.val_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
    train = sessions_to_arrays(train_sessions)
    val = sessions_to_arrays(val_sessions)
    assert plan.target_subject not in set(train["subject_id"])
    assert plan.target_subject not in set(val["subject_id"])

    print_source_sanity(train_sessions, val_sessions)
    print_source_split_sanity(train, val)
    train_x, val_x, preprocess_state = preprocess_source(
        train["x"],
        val["x"],
        robust_clip=args.robust_clip,
    )
    train["x"] = train_x
    val["x"] = val_x
    print_nan_inf_after_preprocessing(("train", train), ("val", val))

    # Target labels are loaded after source-only preprocessing state is fixed;
    # they are used only for final evaluation and saved prediction diagnostics.
    test_sessions = load_seedvig_file_pairs(plan.test_pairs, label_mode=args.label_mode, bandpass=args.bandpass)
    test = sessions_to_arrays(test_sessions)
    assert set(test["subject_id"]) == {plan.target_subject}
    test["x"] = preprocess_target(test["x"], preprocess_state)
    print_target_sanity(test_sessions, test)
    print_nan_inf_after_preprocessing(("test", test))

    train_loader = make_loader(train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test, args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_weights = compute_class_weights(train["y"], args.class_balance)
    print_class_balance(train["y"], args.class_balance, class_weights)
    criterion_weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = EEGNet(channels=17, samples=1600, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0
    best_balanced_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_logits, val_y = predict_logits(model, val_loader, device)
        val_pred = val_logits.argmax(axis=1)
        val_metrics = classification_metrics(val_y, val_pred)
        improved = (
            val_metrics["macro_f1"] > best_macro_f1
            or (
                np.isclose(val_metrics["macro_f1"], best_macro_f1)
                and val_metrics["balanced_accuracy"] > best_balanced_acc
            )
        )
        if improved:
            best_epoch = epoch
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_balanced_acc = float(val_metrics["balanced_accuracy"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"target_subject={plan.target_subject} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a best model")
    model.load_state_dict(best_state)

    test_logits, test_y = predict_logits(model, test_loader, device)
    test_probs = softmax(test_logits)
    test_pred = test_probs.argmax(axis=1)
    test_metrics = classification_metrics(test_y, test_pred)
    print_final_metrics(test_metrics)

    save_predictions(plan.prediction_path, test, test_y, test_pred, test_probs)
    model_config = {"channels": 17, "samples": 1600, "num_classes": 2}
    checkpoint = {
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "command": plan.command,
        "model_state_dict": best_state,
        "model_config": model_config,
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
        "best_epoch": best_epoch,
        "best_val_metric": {
            "macro_f1": best_macro_f1,
            "balanced_accuracy": best_balanced_acc,
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
        best_epoch=best_epoch,
        best_macro_f1=best_macro_f1,
        best_balanced_acc=best_balanced_acc,
    )
    write_checkpoint_manifest_row(
        plan.manifest_path,
        args,
        plan,
        status="success",
        best_epoch=best_epoch,
        best_val_metric=best_macro_f1,
        error="",
    )
    print(f"Saved predictions: {plan.prediction_path}")
    print(f"Saved checkpoint: {plan.checkpoint_path}")
    if plan.latest_checkpoint_path is not None:
        print(f"Saved latest checkpoint: {plan.latest_checkpoint_path}")
    print(f"Saved summary: {plan.summary_path}")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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


def preprocess_source(train_x: np.ndarray, val_x: np.ndarray, *, robust_clip: bool):
    clip_bounds = None
    if robust_clip:
        lo, hi = compute_robust_clip_bounds(train_x)
        train_x = apply_robust_clip(train_x, lo, hi)
        val_x = apply_robust_clip(val_x, lo, hi)
        clip_bounds = (lo, hi)

    mean, std = compute_channel_stats(train_x)
    state = {"clip_bounds": clip_bounds, "mean": mean, "std": std}
    return apply_channel_zscore(train_x, mean, std), apply_channel_zscore(val_x, mean, std), state


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


def make_loader(arrays: dict[str, np.ndarray], batch_size: int, *, shuffle: bool, num_workers: int) -> DataLoader:
    x = torch.from_numpy(arrays["x"]).float().unsqueeze(1)
    y = torch.from_numpy(arrays["y"]).long()
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
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
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        total_count += len(y)
    return total_loss / max(total_count, 1)


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
    best_epoch: int,
    best_macro_f1: float,
    best_balanced_acc: float,
) -> None:
    row = {
        **base_summary_fields(args, plan),
        "status": "success",
        "error": "",
        "train_count": plan.train_counts["usable"],
        "val_count": plan.val_counts["usable"],
        "test_count": plan.test_counts["usable"],
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
        "best_val_metric": best_macro_f1,
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    upsert_summary_row(path, row)


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
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
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
        "n_train_subjects": len(plan.train_subject_ids),
        "n_val_subjects": len(plan.val_subject_ids),
    }


def upsert_summary_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_cols = ["label_mode", "target_subject", "seed", "class_balance"]
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
    print(f"split_counts train={len(train['y'])} val={len(val['y'])}")
    print(
        "split_subjects "
        f"train={sorted(int(s) for s in set(train['subject_id']))} "
        f"val={sorted(int(s) for s in set(val['subject_id']))}"
    )
    for name, arrays in (("train", train), ("val", val)):
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
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "precision", "recall"):
        print(f"  {key}: {metrics[key]:.4f}")
    print("confusion_matrix")
    print(metrics["confusion_matrix"])


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


def print_recommended_gpu_command() -> None:
    print("recommended_later_gpu_command")
    print(
        "python train_eegnet_source.py --run-all-loso --epochs 100 --batch-size 64 "
        "--device cuda --label-mode threshold35 --class-balance weighted_loss"
    )


if __name__ == "__main__":
    main()
