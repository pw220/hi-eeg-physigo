from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import EEGDataset, EEGFold


class StandardDataset(EEGDataset):
    """DrowEEG standard array dataset.

    Required arrays:
    - X: float32, shape (N, C, T)
    - subjects: shape (N,)
    - y: int64, shape (N,), required for current source-only supervised training
    """

    name = "standard-npz"
    label_protocol = "standard"

    def __init__(self, path: str | Path | None = None, **_: object) -> None:
        self.path = None if path is None else Path(path)
        self._arrays: dict[str, np.ndarray] | None = None
        self._metadata: dict[str, Any] = {}
        self.input_channels = 0
        self.input_samples = 0
        self.num_classes = 2

    @classmethod
    def from_arrays(
        cls,
        *,
        X: np.ndarray,
        y: np.ndarray | None = None,
        subjects: np.ndarray,
        sessions: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
        perclos: np.ndarray | None = None,
        sfreq: float | None = None,
        channel_names: list[str] | None = None,
        label_names: dict[int, str] | None = None,
        metadata: dict[str, Any] | None = None,
        extra_arrays: dict[str, np.ndarray] | None = None,
    ) -> "StandardDataset":
        dataset = cls()
        dataset._arrays, dataset._metadata = standardize_arrays(
            X=X,
            y=y,
            subjects=subjects,
            sessions=sessions,
            sample_ids=sample_ids,
            perclos=perclos,
            sfreq=sfreq,
            channel_names=channel_names,
            label_names=label_names,
            metadata=metadata,
            extra_arrays=extra_arrays,
        )
        dataset._sync_shape_metadata()
        return dataset

    def load(self) -> "StandardDataset":
        if self.path is None:
            if self._arrays is None:
                raise ValueError("StandardDataset has no path and no arrays")
            return self
        self._arrays, self._metadata = load_standard_dataset(self.path)
        self._sync_shape_metadata()
        return self

    def get_subjects(self) -> list[int]:
        arrays = self.get_data()
        return sorted({int(subject) for subject in arrays["subject_id"]})

    def get_data(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self.load()
        assert self._arrays is not None
        return self._arrays

    @property
    def X(self) -> np.ndarray:
        return self.get_data()["x"]

    @property
    def y(self) -> np.ndarray:
        arrays = self.get_data()
        if "y" not in arrays:
            raise AttributeError("This standard-npz dataset has no y labels")
        return arrays["y"]

    @property
    def subjects(self) -> np.ndarray:
        return self.get_data()["subject_id"]

    def get_metadata(self) -> dict[str, Any]:
        arrays = self.get_data()
        metadata = self._metadata.get("metadata", {})
        label_distribution = {}
        if "y" in arrays and arrays["y"] is not None:
            values, counts = np.unique(arrays["y"], return_counts=True)
            label_distribution = {int(value): int(count) for value, count in zip(values, counts)}
        return {
            **super().get_metadata(),
            "dataset_name": metadata.get("dataset_name", self.name),
            "protocol_name": metadata.get("protocol_name"),
            "label_mode": metadata.get("label_mode"),
            "path": None if self.path is None else str(self.path),
            "samples": int(len(arrays["x"])),
            "subjects": self.get_subjects(),
            "sessions": sorted(str(session) for session in set(arrays["session_id"].tolist())),
            "label_distribution": label_distribution,
            "sfreq": self._metadata.get("sfreq"),
            "channel_names": self._metadata.get("channel_names"),
            "label_names": self._metadata.get("label_names"),
            "metadata": metadata,
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

    def save(self, path: str | Path) -> None:
        arrays = self.get_data()
        save_standard_dataset(
            path,
            X=arrays["x"],
            y=arrays.get("y"),
            subjects=arrays["subject_id"],
            sessions=arrays.get("session_id"),
            sample_ids=arrays.get("sample_id"),
            perclos=arrays.get("perclos_value"),
            sfreq=self._metadata.get("sfreq"),
            channel_names=self._metadata.get("channel_names"),
            label_names=self._metadata.get("label_names"),
            metadata=self._metadata.get("metadata"),
            extra_arrays=standard_extra_arrays(arrays),
        )

    def to_standard_dataset(self) -> "StandardDataset":
        return self

    def _sync_shape_metadata(self) -> None:
        arrays = self.get_data()
        self.input_channels = int(arrays["x"].shape[1])
        self.input_samples = int(arrays["x"].shape[2])
        if "y" in arrays and arrays["y"] is not None:
            labels = np.unique(arrays["y"])
            self.num_classes = int(np.max(labels)) + 1 if len(labels) else 0


def standardize_arrays(
    *,
    X: np.ndarray,
    y: np.ndarray | None = None,
    subjects: np.ndarray,
    sessions: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    perclos: np.ndarray | None = None,
    sfreq: float | None = None,
    channel_names: list[str] | None = None,
    label_names: dict[int, str] | None = None,
    metadata: dict[str, Any] | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must have shape (N, C, T), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf")
    n_samples = X.shape[0]
    subject_id = _as_1d(subjects, "subjects", n_samples).astype(np.int64)
    if y is None:
        labels = None
    else:
        labels = _as_1d(y, "y", n_samples)
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.all(np.equal(labels, labels.astype(np.int64))):
                raise ValueError("y must contain integer class IDs")
        labels = labels.astype(np.int64)
    session_id = (
        np.asarray([f"session_{int(subject)}" for subject in subject_id], dtype=object)
        if sessions is None
        else _as_1d(sessions, "sessions", n_samples).astype(object)
    )
    sample_id = (
        np.asarray([f"sample_{idx:06d}" for idx in range(n_samples)], dtype=object)
        if sample_ids is None
        else _as_1d(sample_ids, "sample_ids", n_samples).astype(object)
    )
    perclos_value = (
        np.full(n_samples, np.nan, dtype=np.float32)
        if perclos is None
        else _as_1d(perclos, "perclos", n_samples).astype(np.float32)
    )
    arrays = {
        "x": X.astype(np.float32, copy=False),
        "subject_id": subject_id,
        "sample_id": sample_id,
        "session_id": session_id,
        "file_name": np.asarray(["standard-npz"] * n_samples, dtype=object),
        "window_id": np.arange(n_samples, dtype=np.int64),
        "perclos_value": perclos_value,
        "label_mode": np.asarray(["standard"] * n_samples, dtype=object),
        "is_valid_binary_sample": np.ones(n_samples, dtype=bool),
    }
    if labels is not None:
        arrays["y"] = labels
        arrays["label"] = labels
    if extra_arrays is not None:
        for key, values in extra_arrays.items():
            if key in arrays or key in {"X", "subjects", "sessions", "sample_ids", "y"}:
                raise ValueError(f"{key} is reserved and cannot be used as an extra array")
            arrays[key] = _as_1d(values, key, n_samples)
    meta = {
        "sfreq": None if sfreq is None else float(sfreq),
        "channel_names": channel_names,
        "label_names": label_names,
        "metadata": metadata or {},
    }
    return arrays, meta


def save_standard_dataset(
    path: str | Path,
    *,
    X: np.ndarray,
    y: np.ndarray | None = None,
    subjects: np.ndarray,
    sessions: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    perclos: np.ndarray | None = None,
    sfreq: float | None = None,
    channel_names: list[str] | None = None,
    label_names: dict[int, str] | None = None,
    metadata: dict[str, Any] | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    arrays, meta = standardize_arrays(
        X=X,
        y=y,
        subjects=subjects,
        sessions=sessions,
        sample_ids=sample_ids,
        perclos=perclos,
        sfreq=sfreq,
        channel_names=channel_names,
        label_names=label_names,
        metadata=metadata,
        extra_arrays=extra_arrays,
    )
    payload = {
        "X": arrays["x"],
        "subjects": arrays["subject_id"],
        "sessions": arrays["session_id"],
        "sample_ids": arrays["sample_id"],
        "perclos": arrays["perclos_value"],
    }
    if y is not None:
        payload["y"] = arrays["y"]
    if meta["sfreq"] is not None:
        payload["sfreq"] = np.asarray(meta["sfreq"], dtype=np.float32)
    if channel_names is not None:
        payload["channel_names"] = np.asarray(channel_names, dtype=object)
    if label_names is not None:
        payload["label_names_json"] = np.asarray(json.dumps(label_names), dtype=object)
    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(metadata), dtype=object)
    if extra_arrays is not None:
        for key in extra_arrays:
            payload[key] = arrays[key]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_standard_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        if "X" not in data or "subjects" not in data:
            raise ValueError("standard-npz requires at least X and subjects arrays")
        X = data["X"]
        y = data["y"] if "y" in data else None
        subjects = data["subjects"]
        sessions = data["sessions"] if "sessions" in data else None
        sample_ids = data["sample_ids"] if "sample_ids" in data else None
        perclos = data["perclos"] if "perclos" in data else None
        sfreq = float(np.asarray(data["sfreq"]).item()) if "sfreq" in data else None
        channel_names = data["channel_names"].astype(str).tolist() if "channel_names" in data else None
        label_names = _json_scalar(data["label_names_json"]) if "label_names_json" in data else None
        metadata = _json_scalar(data["metadata_json"]) if "metadata_json" in data else None
        reserved = {
            "X",
            "y",
            "subjects",
            "sessions",
            "sample_ids",
            "perclos",
            "sfreq",
            "channel_names",
            "label_names_json",
            "metadata_json",
        }
        extra_arrays = {key: data[key] for key in data.files if key not in reserved}
    if label_names is not None:
        label_names = {int(k): v for k, v in label_names.items()}
    return standardize_arrays(
        X=X,
        y=y,
        subjects=subjects,
        sessions=sessions,
        sample_ids=sample_ids,
        perclos=perclos,
        sfreq=sfreq,
        channel_names=channel_names,
        label_names=label_names,
        metadata=metadata,
        extra_arrays=extra_arrays,
    )


def standard_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    y = arrays.get("y")
    if y is None:
        alert = fatigue = 0
    else:
        counts = np.bincount(y.astype(np.int64), minlength=2)
        alert = int(counts[0])
        fatigue = int(counts[1])
    sessions = arrays.get("session_id")
    session_count = len(set(sessions.tolist())) if sessions is not None else len(set(arrays["subject_id"].tolist()))
    return {
        "sessions": session_count,
        "usable": int(len(arrays["x"])),
        "alert": alert,
        "fatigue": fatigue,
        "excluded": 0,
    }


def make_toy_dataset(
    *,
    n_subjects: int = 4,
    samples_per_subject: int = 20,
    channels: int = 8,
    samples: int = 128,
    sfreq: float = 128.0,
    random_state: int = 42,
) -> StandardDataset:
    """Create a tiny synthetic DrowEEG-standard dataset for examples and smoke tests.

    The toy data is not a benchmark. It only illustrates the required `(N, C, T)`
    array convention, subject-wise LOSO splitting, and `.npz` save/load workflow.
    """
    if n_subjects < 2:
        raise ValueError("n_subjects must be at least 2 for subject-wise evaluation")
    if samples_per_subject < 2:
        raise ValueError("samples_per_subject must be at least 2")
    if channels < 1 or samples < 1:
        raise ValueError("channels and samples must be positive")

    rng = np.random.default_rng(random_state)
    n_samples = int(n_subjects * samples_per_subject)
    subject_ids = np.repeat(np.arange(1, n_subjects + 1, dtype=np.int64), samples_per_subject)
    y = np.tile(np.asarray([0, 1], dtype=np.int64), int(np.ceil(n_samples / 2)))[:n_samples]
    X = rng.normal(0.0, 1.0, size=(n_samples, channels, samples)).astype(np.float32)

    time = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float32)
    class_signal = np.sin(2.0 * np.pi * 8.0 * time).astype(np.float32)
    X[y == 1, 0, :] += 0.35 * class_signal
    if channels > 1:
        X[y == 0, 1, :] += 0.20 * np.cos(2.0 * np.pi * 6.0 * time).astype(np.float32)

    sessions = np.asarray([f"subject_{subject}_session_1" for subject in subject_ids], dtype=object)
    sample_ids = np.asarray([f"toy_s{subject:02d}_w{idx:04d}" for idx, subject in enumerate(subject_ids)], dtype=object)
    channel_names = [f"Ch{idx + 1}" for idx in range(channels)]
    return StandardDataset.from_arrays(
        X=X,
        y=y,
        subjects=subject_ids,
        sessions=sessions,
        sample_ids=sample_ids,
        sfreq=sfreq,
        channel_names=channel_names,
        label_names={0: "alert", 1: "fatigue"},
        metadata={
            "dataset_name": "toy",
            "protocol_name": "toy-binary",
            "description": "Synthetic DrowEEG-standard toy dataset for API examples and smoke tests.",
            "normalization": "not_applied_in_cache",
            "robust_clipping": "not_applied_in_cache",
        },
    )


def _as_1d(values: np.ndarray, name: str, n_samples: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) != n_samples:
        raise ValueError(f"{name} must have shape ({n_samples},), got {values.shape}")
    return values


def _json_scalar(value: np.ndarray) -> Any:
    raw = np.asarray(value).item()
    return json.loads(str(raw))


def standard_extra_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reserved = {
        "x",
        "y",
        "label",
        "subject_id",
        "sample_id",
        "session_id",
        "file_name",
        "window_id",
        "perclos_value",
        "label_mode",
        "is_valid_binary_sample",
    }
    return {key: value for key, value in arrays.items() if key not in reserved}
