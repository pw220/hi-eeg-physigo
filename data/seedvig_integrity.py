from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.seedvig_dataset import (
    _ensure_time_by_channels,
    _extract_eeg_data,
    _extract_eeg_object,
    _extract_labels,
    _extract_sample_rate,
    LabelMode,
    binarize_perclos,
    parse_subject_id,
    sio,
)


@dataclass(frozen=True)
class IntegritySession:
    subject_id: int | None
    session_id: str
    raw_path: Path | None
    label_path: Path | None
    included: bool
    exclusion_reason: str
    raw_exists: bool
    label_exists: bool
    sample_rate: int | None
    eeg_shape: str
    label_length: int | None
    total_windows_before_filtering: int | None
    usable_binary_samples: int
    alert_count: int
    fatigue_count: int
    excluded_count: int
    nan_count: int | None
    inf_count: int | None


@dataclass(frozen=True)
class IntegrityReport:
    sessions: list[IntegritySession]
    total_subjects: int
    total_sessions: int
    included_subject_ids: list[int]
    excluded_subject_ids: list[int]
    label_rule: str
    label_mode: str
    min_class_samples: int

    @property
    def valid_file_pairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        for session in self.sessions:
            if session.included and session.raw_path is not None and session.label_path is not None:
                pairs.append((session.raw_path, session.label_path))
        return pairs


def build_seedvig_integrity_report(
    data_root: str | Path | None = None,
    *,
    raw_data_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    sample_rate: int = 200,
    window_seconds: int = 8,
    label_mode: LabelMode = "threshold35",
    min_class_samples: int = 1,
    metadata_only: bool = False,
) -> IntegrityReport:
    if (raw_data_dir is None) != (label_dir is None):
        raise ValueError("raw_data_dir and label_dir must be provided together")
    if raw_data_dir is None:
        if data_root is None:
            raise ValueError("Provide either data_root or both raw_data_dir and label_dir")
        data_root = Path(data_root)
        raw_dir = data_root / "Raw_Data"
        label_dir = data_root / "perclos_labels"
    else:
        raw_dir = Path(raw_data_dir)
        label_dir = Path(label_dir)
    raw_by_stem = {path.stem: path for path in raw_dir.glob("*.mat")}
    label_by_stem = {path.stem: path for path in label_dir.glob("*.mat")}
    stems = sorted(set(raw_by_stem) | set(label_by_stem))
    if not stems:
        raise FileNotFoundError(f"No SEED-VIG .mat files found under {data_root}")

    sessions = [
        inspect_seedvig_session_integrity(
            stem,
            raw_by_stem.get(stem),
            label_by_stem.get(stem),
            sample_rate=sample_rate,
            window_seconds=window_seconds,
            label_mode=label_mode,
            min_class_samples=min_class_samples,
            metadata_only=metadata_only,
        )
        for stem in stems
    ]
    all_subjects = sorted({s.subject_id for s in sessions if s.subject_id is not None})
    included_subjects = sorted({s.subject_id for s in sessions if s.included and s.subject_id is not None})
    excluded_subjects = sorted(set(all_subjects) - set(included_subjects))
    return IntegrityReport(
        sessions=sessions,
        total_subjects=len(all_subjects),
        total_sessions=len(sessions),
        included_subject_ids=included_subjects,
        excluded_subject_ids=excluded_subjects,
        label_rule=label_rule_for_mode(label_mode),
        label_mode=label_mode,
        min_class_samples=min_class_samples,
    )


def inspect_seedvig_session_integrity(
    stem: str,
    raw_path: Path | None,
    label_path: Path | None,
    *,
    sample_rate: int,
    window_seconds: int,
    label_mode: LabelMode,
    min_class_samples: int,
    metadata_only: bool = False,
) -> IntegritySession:
    subject_id = _safe_subject_id(stem, raw_path, label_path)
    if raw_path is None:
        return _excluded(stem, subject_id, raw_path, label_path, "missing raw EEG")
    if label_path is None:
        return _excluded(stem, subject_id, raw_path, label_path, "missing label file")

    try:
        label_mat = sio.loadmat(label_path, squeeze_me=True, struct_as_record=False)
        perclos = _extract_labels(label_mat).astype(np.float32, copy=False).reshape(-1)
        if metadata_only:
            eeg = None
            detected_sample_rate = sample_rate
            raw_windows = 885
            eeg_shape = "(metadata_only)"
            nan_count = 0
            inf_count = 0
        else:
            raw_mat = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
            eeg_obj = _extract_eeg_object(raw_mat)
            eeg = _ensure_time_by_channels(_extract_eeg_data(eeg_obj), raw_path)
            detected_sample_rate = _extract_sample_rate(raw_mat, eeg_obj)
            window_samples = sample_rate * window_seconds
            raw_windows = int(eeg.shape[0] // window_samples)
            eeg_shape = str(tuple(eeg.shape))
            nan_count = int(np.isnan(eeg).sum())
            inf_count = int(np.isinf(eeg).sum())

        label_result = binarize_perclos(perclos, label_mode=label_mode)
        alert_count = label_result.alert_count
        fatigue_count = label_result.fatigue_count
        usable_count = int(label_result.valid_mask.sum())
        excluded_count = label_result.excluded_count

        reason = ""
        if detected_sample_rate != sample_rate:
            reason = f"other reason: sample_rate {detected_sample_rate} != {sample_rate}"
        elif eeg is not None and eeg.shape[1] != 17:
            reason = f"other reason: expected 17 EEG channels, found {eeg.shape[1]}"
        elif nan_count or inf_count:
            reason = f"other reason: raw EEG contains {nan_count} NaN and {inf_count} Inf values"
        elif raw_windows != 885:
            reason = f"other reason: expected 885 EEG windows, found {raw_windows}"
        elif len(perclos) != 885:
            reason = f"other reason: expected 885 PERCLOS labels, found {len(perclos)}"
        elif raw_windows != len(perclos):
            reason = f"label length mismatch: {raw_windows} windows vs {len(perclos)} labels"
        elif usable_count != int(label_result.valid_mask.sum()):
            reason = "other reason: usable labels do not match valid sample mask"
        elif alert_count < min_class_samples:
            reason = f"too few alert samples: {alert_count} < {min_class_samples}"
        elif fatigue_count < min_class_samples:
            reason = f"too few fatigue samples: {fatigue_count} < {min_class_samples}"

        return IntegritySession(
            subject_id=subject_id,
            session_id=stem,
            raw_path=raw_path,
            label_path=label_path,
            included=not reason,
            exclusion_reason=reason,
            raw_exists=True,
            label_exists=True,
            sample_rate=detected_sample_rate,
            eeg_shape=eeg_shape,
            label_length=int(len(perclos)),
            total_windows_before_filtering=raw_windows,
            usable_binary_samples=usable_count,
            alert_count=alert_count,
            fatigue_count=fatigue_count,
            excluded_count=excluded_count,
            nan_count=nan_count,
            inf_count=inf_count,
        )
    except Exception as exc:  # noqa: BLE001 - integrity report should record failures
        return _excluded(stem, subject_id, raw_path, label_path, f"other reason: {exc}")


def save_integrity_csv(report: IntegrityReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for session in report.sessions:
        rows.append(
            {
                "record_type": "session",
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "included": session.included,
                "exclusion_reason": session.exclusion_reason,
                "raw_exists": session.raw_exists,
                "label_exists": session.label_exists,
                "sample_rate": session.sample_rate,
                "eeg_shape": session.eeg_shape,
                "label_length": session.label_length,
                "label_mode": report.label_mode,
                "total_windows_before_filtering": session.total_windows_before_filtering,
                "usable_binary_samples_after_filtering": session.usable_binary_samples,
                "alert_count": session.alert_count,
                "fatigue_count": session.fatigue_count,
                "excluded_count": session.excluded_count,
                "nan_count": session.nan_count,
                "inf_count": session.inf_count,
            }
        )

    for row in subject_summary_rows(report.sessions):
        rows.append(
            {
                "record_type": "subject",
                "subject_id": row["subject_id"],
                "session_id": "",
                "included": row["included"],
                "exclusion_reason": row["exclusion_reason"],
                "raw_exists": row["raw_exists"],
                "label_exists": row["label_exists"],
                "sample_rate": "",
                "eeg_shape": "",
                "label_length": row["label_length"],
                "label_mode": report.label_mode,
                "total_windows_before_filtering": row["raw_windows_before_label_filtering"],
                "usable_binary_samples_after_filtering": row["usable_binary_samples"],
                "alert_count": row["alert_count"],
                "fatigue_count": row["fatigue_count"],
                "excluded_count": row["excluded_count"],
                "nan_count": row["nan_count"],
                "inf_count": row["inf_count"],
                "included_sessions": row["included_sessions"],
                "excluded_sessions": row["excluded_sessions"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_loso_fold_integrity_report(
    report: IntegrityReport,
    path: str | Path,
    *,
    target_subject: int,
    train_pairs: Iterable[tuple[Path, Path]],
    val_pairs: Iterable[tuple[Path, Path]],
    test_pairs: Iterable[tuple[Path, Path]],
    robust_clip: bool,
    validation_mode: str = "subject_split",
    validation_strategy: str = "deterministic source-subject split controlled by seed and val_subject_ratio",
    val_ratio: float | None = None,
    val_subject_ratio: float | None = None,
    checkpoint_policy: str = "best_val",
    early_stop_enabled: bool = False,
    train_counts: dict[str, int] | None = None,
    val_counts: dict[str, int] | None = None,
    test_counts: dict[str, int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_subjects = sorted({parse_subject_id(raw) for raw, _ in train_pairs})
    val_subjects = sorted({parse_subject_id(raw) for raw, _ in val_pairs})
    test_subjects = sorted({parse_subject_id(raw) for raw, _ in test_pairs})
    session_by_id = {session.session_id: session for session in report.sessions}

    lines = [
        "SEED-VIG LOSO Fold Integrity Report",
        "",
        f"Total subjects: {report.total_subjects}",
        f"Total sessions: {report.total_sessions}",
        f"Included subject IDs: {report.included_subject_ids}",
        f"Excluded subject IDs: {report.excluded_subject_ids}",
        f"Label mode: {report.label_mode}",
        f"Binary label rule: {report.label_rule}",
        "",
        "Excluded sessions:",
    ]
    excluded = [s for s in report.sessions if not s.included]
    if excluded:
        lines.extend(f"- {s.session_id}: {s.exclusion_reason}" for s in excluded)
    else:
        lines.append("- none")

    lines.extend(["", "Subject sample summary:"])
    for row in subject_summary_rows(report.sessions):
        lines.append(
            "- subject {subject_id}: raw_windows={raw_windows_before_label_filtering}, "
            "usable={usable_binary_samples}, alert={alert_count}, fatigue={fatigue_count}, "
            "excluded={excluded_count}, "
            "included_sessions={included_sessions}, excluded_sessions={excluded_sessions}".format(**row)
        )

    lines.extend(
        [
            "",
            "Selected LOSO fold:",
            f"Target subject ID: {target_subject}",
            f"Train subject IDs: {train_subjects}",
            f"Validation subject IDs: {val_subjects}",
            "Validation split strategy: deterministic source-subject split controlled by seed and val_subject_ratio",
            f"Test subject IDs: {test_subjects}",
        ]
    )
    explicit_counts = {"train": train_counts, "val": val_counts, "test": test_counts}
    lines.extend(
        [
            f"Validation mode: {validation_mode}",
            f"Validation strategy: {validation_strategy}",
            f"val_subject_ratio: {val_subject_ratio}",
            f"val_ratio: {val_ratio}",
            f"Checkpoint policy: {checkpoint_policy}",
            f"Early stopping enabled: {early_stop_enabled}",
            "Target labels are audit/evaluation only.",
            "No target samples were used for training, validation, preprocessing statistics, class weights, or model selection.",
        ]
    )
    for name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        counts = explicit_counts[name] if explicit_counts[name] is not None else _counts_for_pairs(pairs, session_by_id)
        lines.append(
            f"{name}: sessions={counts['sessions']}, usable={counts['usable']}, "
            f"alert={counts['alert']}, fatigue={counts['fatigue']}"
        )

    lines.extend(
        [
            "",
            "Preprocessing leakage checks:",
            "Normalization statistics source: source-training data only",
            f"Clipping statistics source: {'source-training data only' if robust_clip else 'not computed; robust clipping disabled'}",
            "Target labels are not used for training, validation, normalization, clipping, or model selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_integrity_report_summary(
    report: IntegrityReport,
    *,
    target_subject: int,
    train_pairs: Iterable[tuple[Path, Path]],
    val_pairs: Iterable[tuple[Path, Path]],
    test_pairs: Iterable[tuple[Path, Path]],
    train_counts: dict[str, int] | None = None,
    val_counts: dict[str, int] | None = None,
    test_counts: dict[str, int] | None = None,
) -> None:
    session_by_id = {session.session_id: session for session in report.sessions}
    print("integrity_report_summary")
    print(f"  total_subjects={report.total_subjects} total_sessions={report.total_sessions}")
    print(f"  label_mode={report.label_mode}")
    print(f"  included_subject_ids={report.included_subject_ids}")
    print(f"  excluded_subject_ids={report.excluded_subject_ids}")
    print(f"  label_rule={report.label_rule}")
    excluded = [s for s in report.sessions if not s.included]
    print(f"  excluded_sessions={len(excluded)}")
    explicit_counts = {"train": train_counts, "val": val_counts, "test": test_counts}
    for name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        counts = explicit_counts[name] if explicit_counts[name] is not None else _counts_for_pairs(pairs, session_by_id)
        print(
            f"  {name}: sessions={counts['sessions']} usable={counts['usable']} "
            f"alert={counts['alert']} fatigue={counts['fatigue']} excluded={counts['excluded']}"
        )
    print(f"  selected_target_subject={target_subject}")
    print("  normalization_stats=source_training_only")


def subject_summary_rows(sessions: Iterable[IntegritySession]) -> list[dict[str, object]]:
    grouped: dict[int, list[IntegritySession]] = defaultdict(list)
    for session in sessions:
        if session.subject_id is not None:
            grouped[session.subject_id].append(session)

    rows = []
    for subject_id in sorted(grouped):
        subject_sessions = grouped[subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "session_id": "",
                "included": all(s.included for s in subject_sessions),
                "exclusion_reason": "; ".join(s.exclusion_reason for s in subject_sessions if s.exclusion_reason),
                "raw_exists": all(s.raw_exists for s in subject_sessions),
                "label_exists": all(s.label_exists for s in subject_sessions),
                "sample_rate": "",
                "eeg_shape": "",
                "label_length": _sum_optional(s.label_length for s in subject_sessions),
                "raw_windows_before_label_filtering": _sum_optional(
                    s.total_windows_before_filtering for s in subject_sessions
                ),
                "usable_binary_samples": sum(s.usable_binary_samples for s in subject_sessions),
                "alert_count": sum(s.alert_count for s in subject_sessions),
                "fatigue_count": sum(s.fatigue_count for s in subject_sessions),
                "excluded_count": sum(s.excluded_count for s in subject_sessions),
                "nan_count": _sum_optional(s.nan_count for s in subject_sessions),
                "inf_count": _sum_optional(s.inf_count for s in subject_sessions),
                "included_sessions": sum(1 for s in subject_sessions if s.included),
                "excluded_sessions": sum(1 for s in subject_sessions if not s.included),
            }
        )
    return rows


def _counts_for_pairs(pairs: Iterable[tuple[Path, Path]], session_by_id: dict[str, IntegritySession]) -> dict[str, int]:
    counter = Counter()
    for raw_path, _ in pairs:
        session = session_by_id[raw_path.stem]
        counter["sessions"] += 1
        counter["usable"] += session.usable_binary_samples
        counter["alert"] += session.alert_count
        counter["fatigue"] += session.fatigue_count
        counter["excluded"] += session.excluded_count
    return dict(counter)


def label_rule_for_mode(label_mode: LabelMode) -> str:
    return binarize_perclos(np.array([0.0], dtype=np.float32), label_mode=label_mode).label_rule


def _excluded(stem, subject_id, raw_path, label_path, reason) -> IntegritySession:
    return IntegritySession(
        subject_id=subject_id,
        session_id=stem,
        raw_path=raw_path,
        label_path=label_path,
        included=False,
        exclusion_reason=reason,
        raw_exists=raw_path is not None,
        label_exists=label_path is not None,
        sample_rate=None,
        eeg_shape="",
        label_length=None,
        total_windows_before_filtering=None,
        usable_binary_samples=0,
        alert_count=0,
        fatigue_count=0,
        excluded_count=0,
        nan_count=None,
        inf_count=None,
    )


def _safe_subject_id(stem: str, raw_path: Path | None, label_path: Path | None) -> int | None:
    try:
        return parse_subject_id(raw_path or label_path or Path(stem))
    except ValueError:
        return None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return int(sum(values))
