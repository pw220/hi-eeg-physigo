from __future__ import annotations


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
