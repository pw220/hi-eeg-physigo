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
