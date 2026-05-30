from __future__ import annotations


def resolve_loso_targets(subjects: list[int], *, target_subject: int, run_all_loso: bool, max_folds: int | None) -> list[int]:
    sorted_subjects = sorted(int(subject) for subject in subjects)
    if run_all_loso:
        return sorted_subjects[:max_folds] if max_folds is not None else sorted_subjects
    if target_subject not in sorted_subjects:
        raise ValueError(f"Target subject {target_subject} not found. Available subjects: {sorted_subjects}")
    return [target_subject]
