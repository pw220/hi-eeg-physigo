from __future__ import annotations

from pathlib import Path

import numpy as np

from data.sadt_dataset import load_sadt_arrays, load_sadt_mat, sadt_counts

from .base import EEGDataset, EEGFold


class SADTBalancedDataset(EEGDataset):
    """Processed balanced SADT mini dataset, not the raw/continuous SADT .set dataset."""

    name = "sadt-balanced"
    label_protocol = "rt_binary"
    input_channels = 30
    input_samples = 384
    num_classes = 2

    def __init__(self, path: str | Path = "data/sad-data.mat", **_: object) -> None:
        self.path = Path(path)
        self._arrays: dict[str, np.ndarray] | None = None

    def load(self) -> "SADTBalancedDataset":
        self._arrays = load_sadt_arrays(self.path)
        return self

    def get_subjects(self) -> list[int]:
        arrays = self.get_data()
        return sorted({int(subject) for subject in arrays["subject_id"]})

    def get_data(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self.load()
        assert self._arrays is not None
        return self._arrays

    def get_metadata(self) -> dict[str, object]:
        arrays = self.get_data()
        return {
            **super().get_metadata(),
            "path": str(self.path),
            "samples": int(len(arrays["y"])),
            "subjects": self.get_subjects(),
            "counts": sadt_counts(arrays),
            "description": "Processed balanced SADT mini dataset; labels are 0=alert and 1=fatigue/drowsy.",
        }

    def build_fold(self, target_subject: int, validation_mode: str = "subject_split", seed: int = 42, **kwargs) -> EEGFold:
        from droweeg.protocols.splits import build_array_loso_fold

        arrays = self.get_data()
        fold = build_array_loso_fold(
            arrays,
            target_subject=target_subject,
            validation_mode=validation_mode,
            seed=seed,
            val_ratio=float(kwargs.get("val_ratio", 0.2)),
            val_subject_ratio=float(kwargs.get("val_subject_ratio", 0.2)),
        )
        return EEGFold(
            train_x=fold["train"]["x"],
            train_y=fold["train"]["y"],
            train_subject_id=fold["train"]["subject_id"],
            val_x=None if fold["val"] is None else fold["val"]["x"],
            val_y=None if fold["val"] is None else fold["val"]["y"],
            val_subject_id=None if fold["val"] is None else fold["val"]["subject_id"],
            test_x=fold["test"]["x"],
            test_y=fold["test"]["y"],
            test_subject_id=fold["test"]["subject_id"],
            sample_id=fold["test"]["sample_id"],
            label_protocol=self.label_protocol,
            input_channels=self.input_channels,
            input_samples=self.input_samples,
            num_classes=self.num_classes,
        )


def inspect(path: str | Path) -> dict[str, object]:
    mat = load_sadt_mat(path)
    dataset = SADTBalancedDataset(path).load()
    arrays = dataset.get_data()
    x = arrays["x"]
    y = arrays["y"]
    subject_id = arrays["subject_id"]
    labels, label_counts = np.unique(y, return_counts=True)
    subjects, subject_counts = np.unique(subject_id, return_counts=True)
    return {
        "path": str(path),
        "variables": sorted(k for k in mat if not k.startswith("__")),
        "EEGsample_shape": tuple(x.shape),
        "substate_shape": tuple(np.asarray(mat["substate"]).shape),
        "subindex_shape": tuple(np.asarray(mat["subindex"]).shape),
        "label_distribution": {int(k): int(v) for k, v in zip(labels, label_counts, strict=True)},
        "subject_distribution": {int(k): int(v) for k, v in zip(subjects, subject_counts, strict=True)},
        "per_subject_label_distribution": _per_subject_label_distribution(y, subject_id),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _per_subject_label_distribution(y: np.ndarray, subject_id: np.ndarray) -> dict[int, dict[int, int]]:
    out = {}
    for subject in sorted({int(s) for s in subject_id}):
        labels, counts = np.unique(y[subject_id == subject], return_counts=True)
        out[subject] = {int(k): int(v) for k, v in zip(labels, counts, strict=True)}
    return out
