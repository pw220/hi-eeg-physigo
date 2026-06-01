from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from data.seedvig_dataset import parse_subject_id
from data.sadt_dataset import sadt_counts
from droweeg.datasets.standard_npz import standard_counts


def resolve_loso_targets(
    subjects: list[int],
    *,
    target_subject: int | None = None,
    target_subjects: list[int] | None = None,
    run_all_loso: bool,
    max_folds: int | None,
) -> list[int]:
    sorted_subjects = sorted(int(subject) for subject in subjects)
    if run_all_loso:
        return sorted_subjects[:max_folds] if max_folds is not None else sorted_subjects
    if target_subjects is not None:
        unknown = [subject for subject in target_subjects if subject not in sorted_subjects]
        if unknown:
            raise ValueError(f"Target subjects {unknown} not found. Available subjects: {sorted_subjects}")
        return list(target_subjects)
    if target_subject is None:
        raise ValueError("target_subject is required when run_all_loso=False")
    if target_subject not in sorted_subjects:
        raise ValueError(f"Target subject {target_subject} not found. Available subjects: {sorted_subjects}")
    return [target_subject]


def subject_mapping_display(mapping: dict[int, object], subjects: list[int]) -> str:
    return ", ".join(f"{subject}(raw={mapping.get(subject, subject)})" for subject in subjects)


def plan_seedvig_splits(args, context, target_subject: int):
    if context.integrity_report is None:
        raise ValueError("SEED-VIG planning requires an integrity report")
    file_pairs = context.file_pairs
    source_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) != target_subject]
    test_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) == target_subject]
    if args.validation_mode == "subject_split":
        train_pairs, val_pairs, test_pairs = split_loso_file_pairs(
            file_pairs,
            target_subject=target_subject,
            val_subject_ratio=args.val_subject_ratio,
            seed=args.seed,
        )
        train_counts = counts_for_pairs(train_pairs, context.integrity_report)
        val_counts = counts_for_pairs(val_pairs, context.integrity_report)
        train_subject_ids = pair_subjects(train_pairs)
        val_subject_ids = pair_subjects(val_pairs)
    elif args.validation_mode == "sample_stratified":
        if not source_pairs or not test_pairs:
            raise ValueError("Invalid LOSO split produced an empty source or test partition")
        train_pairs = source_pairs
        val_pairs = []
        source_counts = counts_for_pairs(source_pairs, context.integrity_report)
        train_counts, val_counts = stratified_metadata_counts(source_counts, args.val_ratio)
        train_subject_ids = pair_subjects(source_pairs)
        val_subject_ids = pair_subjects(source_pairs)
    elif args.validation_mode == "none":
        if not source_pairs or not test_pairs:
            raise ValueError("Invalid LOSO split produced an empty source or test partition")
        train_pairs = source_pairs
        val_pairs = []
        train_counts = counts_for_pairs(source_pairs, context.integrity_report)
        val_counts = zero_counts()
        train_subject_ids = pair_subjects(source_pairs)
        val_subject_ids = []
    else:
        raise ValueError(f"Unsupported validation mode: {args.validation_mode}")
    test_counts = counts_for_pairs(test_pairs, context.integrity_report)
    return train_pairs, val_pairs, test_pairs, train_counts, val_counts, test_counts, train_subject_ids, val_subject_ids


def plan_array_splits(args, context, target_subject: int):
    if context.sadt_arrays is None:
        raise ValueError("Array dataset planning requires loaded arrays")
    arrays = context.sadt_arrays
    source = subset_by_subject(arrays, target_subject, include=False)
    test = subset_by_subject(arrays, target_subject, include=True)
    if len(test["y"]) == 0:
        raise ValueError(f"Target subject {target_subject} has no samples")
    source_subjects = sorted({int(subject) for subject in source["subject_id"]})
    if args.validation_mode == "subject_split":
        train_subject_ids, val_subject_ids = split_subject_ids(source_subjects, args.val_subject_ratio, args.seed)
        train = subset_by_subject_ids(source, train_subject_ids)
        val = subset_by_subject_ids(source, val_subject_ids)
        train_counts = array_counts(train, context)
        val_counts = array_counts(val, context)
    elif args.validation_mode == "sample_stratified":
        train, val = split_arrays_stratified(source, val_ratio=args.val_ratio, seed=args.seed)
        train_subject_ids = source_subjects
        val_subject_ids = source_subjects
        train_counts = array_counts(train, context)
        val_counts = array_counts(val, context)
    elif args.validation_mode == "none":
        train_subject_ids = source_subjects
        val_subject_ids = []
        train_counts = array_counts(source, context)
        val_counts = zero_counts()
    else:
        raise ValueError(f"Unsupported validation mode: {args.validation_mode}")
    return train_counts, val_counts, array_counts(test, context), train_subject_ids, val_subject_ids


def array_counts(arrays: dict[str, np.ndarray], context) -> dict[str, int]:
    if context.label_protocol == "rt_binary":
        return sadt_counts(arrays)
    return standard_counts(arrays)


def raw_subject_ids(context, subject_ids: list[int]) -> list[object]:
    return [context.subject_mapping.get(int(subject_id), int(subject_id)) for subject_id in subject_ids]


def validation_strategy_text(validation_mode: str) -> str:
    if validation_mode == "subject_split":
        return "deterministic source-subject split controlled by seed and val_subject_ratio"
    if validation_mode == "sample_stratified":
        return "sample-level stratified validation within source subjects controlled by seed and val_ratio"
    if validation_mode == "none":
        return "no validation set; all non-target source samples used for training"
    raise ValueError(f"Unsupported validation mode: {validation_mode}")


def split_loso_file_pairs(
    file_pairs: list[tuple[Path, Path]],
    *,
    target_subject: int,
    val_subject_ratio: float,
    seed: int,
):
    test_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) == target_subject]
    source_pairs = [(raw, label) for raw, label in file_pairs if parse_subject_id(raw) != target_subject]
    source_subjects = sorted({parse_subject_id(raw) for raw, _ in source_pairs})
    train_subjects, val_subjects = split_subject_ids(source_subjects, val_subject_ratio, seed)
    train_pairs = [(raw, label) for raw, label in source_pairs if parse_subject_id(raw) in set(train_subjects)]
    val_pairs = [(raw, label) for raw, label in source_pairs if parse_subject_id(raw) in set(val_subjects)]
    return train_pairs, val_pairs, test_pairs


def split_subject_ids(subjects: list[int], val_subject_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    subjects = sorted(subjects)
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_subject_ratio)))
    if val_count >= len(shuffled):
        raise ValueError("Validation subject split would consume all source subjects")
    val_subjects = sorted(shuffled[:val_count])
    train_subjects = sorted(shuffled[val_count:])
    return train_subjects, val_subjects


def subset_by_subject(arrays: dict[str, np.ndarray], subject_id: int, *, include: bool) -> dict[str, np.ndarray]:
    mask = arrays["subject_id"] == subject_id
    if not include:
        mask = ~mask
    return {key: value[mask] for key, value in arrays.items()}


def subset_by_subject_ids(arrays: dict[str, np.ndarray], subject_ids: list[int]) -> dict[str, np.ndarray]:
    subject_ids = np.asarray(subject_ids)
    mask = np.isin(arrays["subject_id"], subject_ids)
    return {key: value[mask] for key, value in arrays.items()}


def split_arrays_stratified(arrays: dict[str, np.ndarray], *, val_ratio: float, seed: int):
    labels = arrays["y"]
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in sorted(np.unique(labels).tolist()):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < 2:
            raise ValueError(f"Cannot stratify class {label}: only {len(label_indices)} samples")
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


def pair_subjects(pairs: list[tuple[Path, Path]]) -> list[int]:
    return sorted({parse_subject_id(raw_path) for raw_path, _ in pairs})


def counts_for_pairs(pairs: list[tuple[Path, Path]], report) -> dict[str, int]:
    selected = {str(raw_path) for raw_path, _ in pairs}
    usable = alert = fatigue = excluded = 0
    sessions = 0
    for session in report.sessions:
        if str(session.raw_path) not in selected:
            continue
        sessions += 1
        usable += session.usable_samples
        alert += session.alert_count
        fatigue += session.fatigue_count
        excluded += session.excluded_count
    return {
        "sessions": sessions,
        "usable": usable,
        "alert": alert,
        "fatigue": fatigue,
        "excluded": excluded,
    }


def zero_counts() -> dict[str, int]:
    return {"sessions": 0, "usable": 0, "alert": 0, "fatigue": 0, "excluded": 0}


def stratified_metadata_counts(source_counts: dict[str, int], val_ratio: float) -> tuple[dict[str, int], dict[str, int]]:
    val_alert = int(round(source_counts["alert"] * val_ratio))
    val_fatigue = int(round(source_counts["fatigue"] * val_ratio))
    val_alert = min(max(1, val_alert), max(1, source_counts["alert"] - 1))
    val_fatigue = min(max(1, val_fatigue), max(1, source_counts["fatigue"] - 1))
    val_counts = {
        "sessions": source_counts["sessions"],
        "usable": val_alert + val_fatigue,
        "alert": val_alert,
        "fatigue": val_fatigue,
        "excluded": 0,
    }
    train_counts = {
        "sessions": source_counts["sessions"],
        "usable": source_counts["usable"] - val_counts["usable"],
        "alert": source_counts["alert"] - val_alert,
        "fatigue": source_counts["fatigue"] - val_fatigue,
        "excluded": source_counts["excluded"],
    }
    return train_counts, val_counts


def subject_mapping_from_arrays(arrays: dict[str, np.ndarray]) -> dict[int, object]:
    raw_subjects = arrays.get("subject_id_raw", arrays["subject_id"])
    mapping: dict[int, object] = {}
    for subject_index, raw_subject in zip(arrays["subject_id"], raw_subjects, strict=True):
        mapping.setdefault(int(subject_index), python_scalar(raw_subject))
    return dict(sorted(mapping.items()))


def python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
