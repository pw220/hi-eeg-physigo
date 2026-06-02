from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.seedvig_dataset import (
    _extract_eeg_data,
    _extract_eeg_object,
    _extract_labels,
    _extract_sample_rate,
    parse_subject_id,
)
from eegda.datasets.standard_npz import save_standard_dataset


@dataclass
class SessionCache:
    subject_id: int
    session_id: str
    X: np.ndarray
    y: np.ndarray
    perclos: np.ndarray
    sample_ids: np.ndarray
    raw_windows: int
    usable_samples: int
    alert_count: int
    fatigue_count: int
    discarded_count: int
    min_perclos: float
    max_perclos: float
    mean_perclos: float
    raw_nan_count: int
    raw_inf_count: int
    included: bool = True
    exclusion_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EEGDA-standard SEED-VIG cached datasets.")
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--label-mode", choices=("threshold35", "strict035070"), required=True)
    parser.add_argument("--alert-threshold", type=float, default=0.35)
    parser.add_argument("--fatigue-threshold", type=float, default=0.70)
    parser.add_argument("--window-seconds", type=int, default=8)
    parser.add_argument("--sfreq", type=int, default=200)
    parser.add_argument("--expected-channels", type=int, default=17)
    parser.add_argument("--expected-windows-per-session", type=int, default=885)
    parser.add_argument("--min-samples-per-class", type=int, default=0)
    parser.add_argument("--filter-level", choices=("none", "subject"), default="none")
    parser.add_argument("--session-policy", choices=("all_valid", "one_most_balanced"), default="all_valid")
    parser.add_argument("--balance-mode", choices=("none",), default="none")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    if args.label_mode == "strict035070" and args.alert_threshold >= args.fatigue_threshold:
        raise ValueError("--alert-threshold must be smaller than --fatigue-threshold")

    raw_files = sorted(Path(args.raw_data_dir).glob("*.mat"))
    if not raw_files:
        raise FileNotFoundError(f"No raw .mat files found in {args.raw_data_dir}")

    sessions: list[SessionCache] = []
    missing_label_rows = []
    for raw_path in raw_files:
        label_path = Path(args.label_dir) / raw_path.name
        if not label_path.exists():
            missing_label_rows.append(missing_label_report_row(raw_path))
            continue
        sessions.append(build_session_cache(raw_path, label_path, args))

    if not sessions:
        raise RuntimeError("No valid SEED-VIG sessions were cached.")

    apply_subject_filter(sessions, args)
    apply_session_policy(sessions, args)
    included = [session for session in sessions if session.included]
    if not included:
        raise RuntimeError("All sessions were excluded; no cache was written.")

    X = np.concatenate([session.X for session in included], axis=0)
    y = np.concatenate([session.y for session in included], axis=0)
    subjects = np.concatenate(
        [np.full(session.usable_samples, session.subject_id, dtype=np.int64) for session in included]
    )
    session_ids = np.concatenate(
        [np.full(session.usable_samples, session.session_id, dtype=object) for session in included]
    )
    sample_ids = np.concatenate([session.sample_ids for session in included])
    perclos = np.concatenate([session.perclos for session in included]).astype(np.float32, copy=False)

    metadata = build_metadata(args, output_path, sessions, included, raw_session_count=len(raw_files))
    save_standard_dataset(
        output_path,
        X=X,
        y=y,
        subjects=subjects,
        sessions=session_ids,
        sample_ids=sample_ids,
        perclos=perclos,
        sfreq=args.sfreq,
        channel_names=[f"EEG{i + 1}" for i in range(args.expected_channels)],
        label_names={0: "alert", 1: "fatigue"},
        metadata=metadata,
    )

    report_path = output_path.with_name(f"{output_path.stem}_report.csv")
    report = pd.DataFrame([session_report_row(session) for session in sessions] + missing_label_rows)
    report.sort_values(["subject_id", "session_id"], inplace=True)
    report.to_csv(report_path, index=False)

    print_summary(args, sessions, included, X, y, report_path, output_path, raw_session_count=len(raw_files))


def build_session_cache(raw_path: Path, label_path: Path, args: argparse.Namespace) -> SessionCache:
    raw_mat = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
    label_mat = sio.loadmat(label_path, squeeze_me=True, struct_as_record=False)
    eeg_obj = _extract_eeg_object(raw_mat)
    eeg = _extract_eeg_data(eeg_obj).astype(np.float32, copy=False)
    detected_sfreq = _extract_sample_rate(raw_mat, eeg_obj)
    if detected_sfreq != args.sfreq:
        raise ValueError(f"{raw_path.name}: expected sfreq={args.sfreq}, found {detected_sfreq}")

    eeg = ensure_time_by_channels(eeg, raw_path, args.expected_channels)
    if eeg.shape[1] != args.expected_channels:
        raise ValueError(f"{raw_path.name}: expected {args.expected_channels} channels, found {eeg.shape[1]}")
    raw_nan_count = int(np.isnan(eeg).sum())
    raw_inf_count = int(np.isinf(eeg).sum())
    if raw_nan_count or raw_inf_count:
        raise ValueError(f"{raw_path.name}: raw EEG contains {raw_nan_count} NaN and {raw_inf_count} Inf values")

    perclos_all = _extract_labels(label_mat).astype(np.float32, copy=False).reshape(-1)
    window_samples = args.sfreq * args.window_seconds
    raw_windows = eeg.shape[0] // window_samples
    if raw_windows != args.expected_windows_per_session:
        raise ValueError(
            f"{raw_path.name}: expected {args.expected_windows_per_session} windows, found {raw_windows}"
        )
    if len(perclos_all) != args.expected_windows_per_session:
        raise ValueError(
            f"{label_path.name}: expected {args.expected_windows_per_session} PERCLOS labels, found {len(perclos_all)}"
        )
    if raw_windows != len(perclos_all):
        raise ValueError(f"{raw_path.name}: {raw_windows} EEG windows but {len(perclos_all)} labels")

    eeg = eeg[: raw_windows * window_samples]
    X_all = eeg.reshape(raw_windows, window_samples, args.expected_channels).transpose(0, 2, 1)
    labels, valid_mask = binarize_perclos(
        perclos_all,
        label_mode=args.label_mode,
        alert_threshold=args.alert_threshold,
        fatigue_threshold=args.fatigue_threshold,
    )
    X = np.ascontiguousarray(X_all[valid_mask], dtype=np.float32)
    y = labels.astype(np.int64, copy=False)
    perclos = np.ascontiguousarray(perclos_all[valid_mask], dtype=np.float32)
    subject_id = parse_subject_id(raw_path)
    session_id = raw_path.stem
    window_ids = np.arange(raw_windows, dtype=np.int64)[valid_mask]
    sample_ids = np.asarray([f"{session_id}_w{window_id:04d}" for window_id in window_ids], dtype=object)
    return SessionCache(
        subject_id=subject_id,
        session_id=session_id,
        X=X,
        y=y,
        perclos=perclos,
        sample_ids=sample_ids,
        raw_windows=raw_windows,
        usable_samples=int(len(y)),
        alert_count=int((y == 0).sum()),
        fatigue_count=int((y == 1).sum()),
        discarded_count=int((~valid_mask).sum()),
        min_perclos=float(np.min(perclos_all)),
        max_perclos=float(np.max(perclos_all)),
        mean_perclos=float(np.mean(perclos_all)),
        raw_nan_count=raw_nan_count,
        raw_inf_count=raw_inf_count,
    )


def binarize_perclos(
    perclos: np.ndarray,
    *,
    label_mode: str,
    alert_threshold: float,
    fatigue_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if label_mode == "threshold35":
        valid_mask = np.ones_like(perclos, dtype=bool)
        labels = np.where(perclos <= alert_threshold, 0, 1)[valid_mask]
        return labels.astype(np.int64), valid_mask
    if label_mode == "strict035070":
        alert = perclos <= alert_threshold
        fatigue = perclos >= fatigue_threshold
        valid_mask = alert | fatigue
        labels = np.where(fatigue[valid_mask], 1, 0)
        return labels.astype(np.int64), valid_mask
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def ensure_time_by_channels(eeg: np.ndarray, raw_path: Path, expected_channels: int) -> np.ndarray:
    if eeg.ndim != 2:
        raise ValueError(f"{raw_path.name}: expected 2D EEG array, found shape {eeg.shape}")
    if eeg.shape[1] == expected_channels:
        return eeg
    if eeg.shape[0] == expected_channels:
        return eeg.T
    raise ValueError(
        f"{raw_path.name}: cannot infer {expected_channels}-channel EEG orientation from {eeg.shape}"
    )


def apply_subject_filter(sessions: list[SessionCache], args: argparse.Namespace) -> None:
    if args.filter_level == "none":
        return
    subject_counts: dict[int, np.ndarray] = {}
    for session in sessions:
        counts = subject_counts.setdefault(session.subject_id, np.zeros(2, dtype=np.int64))
        counts[0] += session.alert_count
        counts[1] += session.fatigue_count
    for session in sessions:
        counts = subject_counts[session.subject_id]
        if counts[0] < args.min_samples_per_class:
            session.included = False
            session.exclusion_reason = f"subject_alert_count<{args.min_samples_per_class}"
        elif counts[1] < args.min_samples_per_class:
            session.included = False
            session.exclusion_reason = f"subject_fatigue_count<{args.min_samples_per_class}"


def apply_session_policy(sessions: list[SessionCache], args: argparse.Namespace) -> None:
    if args.session_policy == "all_valid":
        return
    if args.session_policy != "one_most_balanced":
        raise ValueError(f"Unsupported session_policy: {args.session_policy}")

    sessions_by_subject: dict[int, list[SessionCache]] = {}
    for session in sessions:
        if session.included:
            sessions_by_subject.setdefault(session.subject_id, []).append(session)

    for subject_sessions in sessions_by_subject.values():
        if len(subject_sessions) <= 1:
            continue
        selected = sorted(subject_sessions, key=session_balance_key)[0]
        for session in subject_sessions:
            if session is selected:
                continue
            session.included = False
            session.exclusion_reason = "not_selected_by_one_most_balanced_session_policy"


def session_balance_key(session: SessionCache) -> tuple[int, int, int, str]:
    class_imbalance = abs(session.alert_count - session.fatigue_count)
    minority_count = min(session.alert_count, session.fatigue_count)
    total_count = session.alert_count + session.fatigue_count
    return (class_imbalance, -minority_count, -total_count, session.session_id)


def build_metadata(
    args: argparse.Namespace,
    output_path: Path,
    sessions: list[SessionCache],
    included: list[SessionCache],
    *,
    raw_session_count: int,
) -> dict[str, object]:
    return {
        "dataset_name": "seedvig",
        "source_format": "Raw_Data + perclos_labels",
        "protocol_name": output_path.stem,
        "label_mode": args.label_mode,
        "label_rule": label_rule_text(args),
        "alert_threshold": args.alert_threshold,
        "fatigue_threshold": args.fatigue_threshold,
        "window_seconds": args.window_seconds,
        "sfreq": args.sfreq,
        "input_channels": args.expected_channels,
        "input_samples": args.sfreq * args.window_seconds,
        "expected_windows_per_session": args.expected_windows_per_session,
        "filter_level": args.filter_level,
        "min_samples_per_class": args.min_samples_per_class,
        "session_policy": args.session_policy,
        "balance_mode": args.balance_mode,
        "normalization": "not_applied_in_cache",
        "robust_clipping": "not_applied_in_cache",
        "bandpass": "not_applied_in_cache",
        "raw_subject_count": len({session.subject_id for session in sessions}),
        "raw_session_count": raw_session_count,
        "loaded_session_count": len(sessions),
        "included_subject_count": len({session.subject_id for session in included}),
        "included_session_count": len(included),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def label_rule_text(args: argparse.Namespace) -> str:
    if args.label_mode == "threshold35":
        return f"PERCLOS <= {args.alert_threshold} alert(0); PERCLOS > {args.alert_threshold} fatigue(1)"
    return (
        f"PERCLOS <= {args.alert_threshold} alert(0); "
        f"PERCLOS >= {args.fatigue_threshold} fatigue(1); "
        f"{args.alert_threshold} < PERCLOS < {args.fatigue_threshold} discarded"
    )


def session_report_row(session: SessionCache) -> dict[str, object]:
    return {
        "subject_id": session.subject_id,
        "session_id": session.session_id,
        "raw_windows": session.raw_windows,
        "usable_samples": session.usable_samples,
        "alert_count": session.alert_count,
        "fatigue_count": session.fatigue_count,
        "discarded_count": session.discarded_count,
        "min_perclos": session.min_perclos,
        "max_perclos": session.max_perclos,
        "mean_perclos": session.mean_perclos,
        "raw_nan_count": session.raw_nan_count,
        "raw_inf_count": session.raw_inf_count,
        "included": session.included,
        "exclusion_reason": session.exclusion_reason,
    }


def missing_label_report_row(raw_path: Path) -> dict[str, object]:
    subject_id = parse_subject_id(raw_path)
    return {
        "subject_id": subject_id,
        "session_id": raw_path.stem,
        "raw_windows": 0,
        "usable_samples": 0,
        "alert_count": 0,
        "fatigue_count": 0,
        "discarded_count": 0,
        "min_perclos": np.nan,
        "max_perclos": np.nan,
        "mean_perclos": np.nan,
        "raw_nan_count": np.nan,
        "raw_inf_count": np.nan,
        "included": False,
        "exclusion_reason": "missing_label_file",
    }


def print_summary(
    args: argparse.Namespace,
    sessions: list[SessionCache],
    included: list[SessionCache],
    X: np.ndarray,
    y: np.ndarray,
    report_path: Path,
    output_path: Path,
    raw_session_count: int,
) -> None:
    excluded = [session for session in sessions if not session.included]
    included_subjects = {session.subject_id for session in included}
    all_subjects = {session.subject_id for session in sessions}
    subjects_with_excluded_sessions = {session.subject_id for session in excluded}
    print("seedvig_standard_cache_summary")
    print(f"  output_path={output_path}")
    print(f"  report_path={report_path}")
    print(f"  total_raw_subjects={len(all_subjects)}")
    print(f"  total_raw_sessions={raw_session_count}")
    print(f"  loaded_sessions={len(sessions)}")
    print(f"  included_subjects={sorted(included_subjects)}")
    print(f"  included_sessions={len(included)}")
    print(f"  fully_excluded_subjects={sorted(all_subjects - included_subjects)}")
    print(f"  subjects_with_excluded_sessions={sorted(subjects_with_excluded_sessions)}")
    print(f"  excluded_sessions={len(excluded)}")
    print(f"  total_samples={len(y)}")
    print(f"  total_alert_count={int((y == 0).sum())}")
    print(f"  total_fatigue_count={int((y == 1).sum())}")
    print(f"  final_X_shape={X.shape}")
    print(f"  final_nan_count={int(np.isnan(X).sum())}")
    print(f"  final_inf_count={int(np.isinf(X).sum())}")
    print(f"  label_protocol={label_rule_text(args)}")
    print(f"  subject_level_filtering={args.filter_level == 'subject'}")
    print(f"  session_policy={args.session_policy}")
    print(f"  balancing_applied={args.balance_mode != 'none'}")
    print("  normalization=not_applied_in_cache")
    print("  robust_clipping=not_applied_in_cache")
    print("  bandpass=not_applied_in_cache")
    print("per_subject_label_distribution")
    for subject_id in sorted({session.subject_id for session in included}):
        subject_sessions = [session for session in included if session.subject_id == subject_id]
        alert = sum(session.alert_count for session in subject_sessions)
        fatigue = sum(session.fatigue_count for session in subject_sessions)
        print(f"  subject={subject_id} alert={alert} fatigue={fatigue} sessions={len(subject_sessions)}")
    print("per_session_label_distribution")
    for session in included:
        print(
            f"  subject={session.subject_id} session={session.session_id} "
            f"alert={session.alert_count} fatigue={session.fatigue_count} discarded={session.discarded_count}"
        )


if __name__ == "__main__":
    main()
