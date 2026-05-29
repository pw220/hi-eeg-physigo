from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class SadtData:
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    sample_id: np.ndarray
    session_id: np.ndarray
    file_name: np.ndarray
    window_id: np.ndarray
    perclos_value: np.ndarray
    label_mode: np.ndarray
    is_valid_binary_sample: np.ndarray


def load_sadt_mat(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        mat = loadmat(path, squeeze_me=False, struct_as_record=False)
        return {key: value for key, value in mat.items() if not key.startswith("__")}
    except NotImplementedError:
        import h5py

        out = {}
        with h5py.File(path, "r") as handle:
            for key in handle.keys():
                value = np.array(handle[key])
                if value.ndim > 1:
                    value = np.transpose(value)
                out[key] = value
        return out


def load_sadt_arrays(path: str | Path) -> dict[str, np.ndarray]:
    mat = load_sadt_mat(path)
    required = ("EEGsample", "substate", "subindex")
    missing = [key for key in required if key not in mat]
    if missing:
        raise KeyError(f"SADT file is missing required variables: {missing}")

    x = np.asarray(mat["EEGsample"], dtype=np.float32)
    y = np.asarray(mat["substate"]).reshape(-1).astype(np.int64)
    subject_id = np.asarray(mat["subindex"]).reshape(-1).astype(np.int64)
    validate_sadt_arrays(x, y, subject_id)

    n_samples = len(y)
    sample_id = np.array([f"sadt_{idx:04d}" for idx in range(n_samples)], dtype=object)
    return {
        "x": np.ascontiguousarray(x, dtype=np.float32),
        "y": y,
        "subject_id": subject_id,
        "sample_id": sample_id,
        "session_id": np.array([f"subject_{sid}" for sid in subject_id], dtype=object),
        "file_name": np.full(n_samples, Path(path).name, dtype=object),
        "window_id": np.arange(n_samples, dtype=np.int64),
        "perclos_value": np.full(n_samples, np.nan, dtype=np.float32),
        "label_mode": np.full(n_samples, "rt_binary", dtype=object),
        "is_valid_binary_sample": np.ones(n_samples, dtype=bool),
    }


def validate_sadt_arrays(x: np.ndarray, y: np.ndarray, subject_id: np.ndarray) -> None:
    if x.ndim != 3:
        raise ValueError(f"EEGsample must have shape (N, 30, 384); found {x.shape}")
    if x.shape[1:] != (30, 384):
        raise ValueError(f"EEGsample must have shape (N, 30, 384); found {x.shape}")
    if len(y) != x.shape[0] or len(subject_id) != x.shape[0]:
        raise ValueError(
            f"SADT length mismatch: EEGsample={x.shape[0]} substate={len(y)} subindex={len(subject_id)}"
        )
    labels = set(np.unique(y).astype(int).tolist())
    if labels != {0, 1}:
        raise ValueError(f"SADT labels must be exactly {{0, 1}}; found {sorted(labels)}")
    subjects = set(np.unique(subject_id).astype(int).tolist())
    if subjects != set(range(1, 12)):
        raise ValueError(f"SADT subject IDs must be exactly 1..11; found {sorted(subjects)}")
    nan_count = int(np.isnan(x).sum())
    inf_count = int(np.isinf(x).sum())
    if nan_count or inf_count:
        raise ValueError(f"SADT EEGsample contains {nan_count} NaN and {inf_count} Inf values")


def sadt_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    y = arrays["y"].astype(np.int64, copy=False)
    return {
        "sessions": len(set(arrays["subject_id"].astype(int).tolist())),
        "usable": int(len(y)),
        "alert": int((y == 0).sum()),
        "fatigue": int((y == 1).sum()),
        "excluded": 0,
    }


def inspect_sadt(path: str | Path) -> None:
    mat = load_sadt_mat(path)
    print("sadt_inspection")
    print(f"  path={path}")
    print(f"  variables={sorted(mat.keys())}")
    for key, value in mat.items():
        if hasattr(value, "shape"):
            print(f"  {key}_shape={tuple(value.shape)} dtype={value.dtype}")
    arrays = load_sadt_arrays(path)
    x = arrays["x"]
    y = arrays["y"]
    subject_id = arrays["subject_id"]
    print(f"  EEGsample_nan_count={int(np.isnan(x).sum())}")
    print(f"  EEGsample_inf_count={int(np.isinf(x).sum())}")
    print(f"  EEGsample_min={float(np.min(x)):.6f}")
    print(f"  EEGsample_max={float(np.max(x)):.6f}")
    print(f"  EEGsample_mean={float(np.mean(x)):.6f}")
    print(f"  EEGsample_std={float(np.std(x)):.6f}")
    print(f"  label_distribution={_distribution(y)}")
    print(f"  subject_distribution={_distribution(subject_id)}")
    print("  per_subject_label_distribution")
    for sid in sorted(np.unique(subject_id).astype(int).tolist()):
        print(f"    subject_{sid}={_distribution(y[subject_id == sid])}")


def _distribution(values: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(values.astype(int), return_counts=True)
    return {int(key): int(value) for key, value in zip(unique, counts, strict=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SADT processed EEG .mat file")
    parser.add_argument("--sadt-path", default="data/sad-data.mat")
    args = parser.parse_args()
    inspect_sadt(args.sadt_path)


if __name__ == "__main__":
    main()
