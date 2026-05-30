from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def build_array_loso_fold(
    arrays: dict[str, np.ndarray],
    *,
    target_subject: int,
    validation_mode: str,
    seed: int,
    val_ratio: float,
    val_subject_ratio: float,
) -> dict[str, dict[str, np.ndarray] | None]:
    source = _subset_by_subject(arrays, target_subject, include=False)
    test = _subset_by_subject(arrays, target_subject, include=True)
    if validation_mode == "none":
        return {"train": source, "val": None, "test": test}
    if validation_mode == "sample_stratified":
        train_idx, val_idx = next(
            StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed).split(
                np.zeros(len(source["y"])),
                source["y"],
            )
        )
        return {"train": _subset_by_index(source, train_idx), "val": _subset_by_index(source, val_idx), "test": test}
    if validation_mode == "subject_split":
        subjects = sorted({int(subject) for subject in source["subject_id"]})
        rng = np.random.default_rng(seed)
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_subject_ratio)))
        val_subjects = sorted(shuffled[:n_val])
        train_subjects = sorted(shuffled[n_val:])
        return {
            "train": _subset_by_subject_ids(source, train_subjects),
            "val": _subset_by_subject_ids(source, val_subjects),
            "test": test,
        }
    raise ValueError(f"Unsupported validation mode: {validation_mode}")


def _subset_by_subject(arrays: dict[str, np.ndarray], target_subject: int, *, include: bool) -> dict[str, np.ndarray]:
    mask = arrays["subject_id"] == target_subject
    if not include:
        mask = ~mask
    return _subset_by_mask(arrays, mask)


def _subset_by_subject_ids(arrays: dict[str, np.ndarray], subjects: list[int]) -> dict[str, np.ndarray]:
    return _subset_by_mask(arrays, np.isin(arrays["subject_id"], np.asarray(subjects)))


def _subset_by_index(arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[idx] for key, value in arrays.items()}


def _subset_by_mask(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in arrays.items()}
