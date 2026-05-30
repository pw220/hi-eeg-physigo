from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EEGFold:
    train_x: np.ndarray
    train_y: np.ndarray
    train_subject_id: np.ndarray
    val_x: np.ndarray | None
    val_y: np.ndarray | None
    val_subject_id: np.ndarray | None
    test_x: np.ndarray
    test_y: np.ndarray
    test_subject_id: np.ndarray
    sample_id: np.ndarray
    label_protocol: str
    input_channels: int
    input_samples: int
    num_classes: int


class EEGDataset:
    name: str
    label_protocol: str
    input_channels: int
    input_samples: int
    num_classes: int = 2

    @classmethod
    def from_arrays(
        cls,
        *,
        X: np.ndarray,
        y: np.ndarray | None = None,
        subjects: np.ndarray,
        sessions: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
        sfreq: float | None = None,
        channel_names: list[str] | None = None,
        label_names: dict[int, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EEGDataset":
        from droweeg.datasets.standard_npz import StandardDataset

        return StandardDataset.from_arrays(
            X=X,
            y=y,
            subjects=subjects,
            sessions=sessions,
            sample_ids=sample_ids,
            sfreq=sfreq,
            channel_names=channel_names,
            label_names=label_names,
            metadata=metadata,
        )

    def load(self) -> "EEGDataset":
        raise NotImplementedError

    def get_subjects(self) -> list[int]:
        raise NotImplementedError

    def get_data(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label_protocol": self.label_protocol,
            "input_channels": self.input_channels,
            "input_samples": self.input_samples,
            "num_classes": self.num_classes,
        }

    def build_fold(self, target_subject: int, validation_mode: str = "subject_split", seed: int = 42, **kwargs) -> EEGFold:
        raise NotImplementedError

    def to_standard_dataset(self) -> "EEGDataset":
        raise NotImplementedError
