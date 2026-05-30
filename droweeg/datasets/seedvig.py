from __future__ import annotations

from pathlib import Path

from data.seedvig_integrity import build_seedvig_integrity_report
from data.seedvig_dataset import parse_subject_id

from .base import EEGDataset


class SeedVIGDataset(EEGDataset):
    name = "seedvig"
    input_channels = 17
    input_samples = 1600
    num_classes = 2

    def __init__(
        self,
        data_root: str | Path = "data/raw/SEED-VIG",
        raw_data_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        label_mode: str = "threshold35",
        min_class_samples: int = 1,
        **_: object,
    ) -> None:
        self.data_root = data_root
        self.raw_data_dir = raw_data_dir
        self.label_dir = label_dir
        self.label_mode = label_mode
        self.label_protocol = label_mode
        self.min_class_samples = min_class_samples
        self._report = None

    def load(self) -> "SeedVIGDataset":
        self._report = build_seedvig_integrity_report(
            self.data_root,
            raw_data_dir=self.raw_data_dir,
            label_dir=self.label_dir,
            label_mode=self.label_mode,
            min_class_samples=self.min_class_samples,
            metadata_only=True,
        )
        return self

    def get_subjects(self) -> list[int]:
        if self._report is None:
            self.load()
        assert self._report is not None
        return sorted({parse_subject_id(raw_path) for raw_path, _ in self._report.valid_file_pairs})

    def get_data(self):
        if self._report is None:
            self.load()
        return {"integrity_report": self._report}

    def get_metadata(self) -> dict[str, object]:
        if self._report is None:
            self.load()
        assert self._report is not None
        return {
            **super().get_metadata(),
            "data_root": str(self.data_root),
            "label_mode": self.label_mode,
            "subjects": self.get_subjects(),
            "sessions": len(self._report.valid_file_pairs),
            "label_rule": self._report.label_rule,
        }
