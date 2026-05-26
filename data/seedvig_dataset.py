from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Literal

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
    from scipy import signal
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise ImportError(
        "SEED-VIG loading requires scipy. Install project dependencies with "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc


@dataclass(frozen=True)
class SeedVigSession:
    subject_id: int
    session_id: str
    raw_path: Path
    label_path: Path
    label_mode: str
    x: np.ndarray
    y: np.ndarray
    perclos: np.ndarray
    window_ids: np.ndarray
    sample_ids: np.ndarray
    file_names: np.ndarray
    sample_index: pd.DataFrame
    raw_segment_count: int
    dropped_middle_count: int
    alert_count: int
    fatigue_count: int
    nan_count: int
    inf_count: int


@dataclass(frozen=True)
class LabelResult:
    labels: np.ndarray
    valid_mask: np.ndarray
    excluded_count: int
    alert_count: int
    fatigue_count: int
    label_rule: str


LabelMode = Literal["threshold35", "strict035070"]


def parse_subject_id(path: Path) -> int:
    match = re.match(r"^(\d+)_", path.stem)
    if not match:
        raise ValueError(f"Cannot parse subject id from filename: {path.name}")
    return int(match.group(1))


def discover_seedvig_files(
    data_root: str | Path | None = None,
    *,
    raw_data_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
) -> list[tuple[Path, Path]]:
    if (raw_data_dir is None) != (label_dir is None):
        raise ValueError("raw_data_dir and label_dir must be provided together")
    if raw_data_dir is None:
        if data_root is None:
            raise ValueError("Provide either data_root or both raw_data_dir and label_dir")
        data_root = Path(data_root)
        raw_dir = data_root / "Raw_Data"
        label_dir = data_root / "perclos_labels"
    else:
        raw_dir = Path(raw_data_dir)
        label_dir = Path(label_dir)
    raw_files = sorted(raw_dir.glob("*.mat"))
    if not raw_files:
        raise FileNotFoundError(f"No raw .mat files found in {raw_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for raw_path in raw_files:
        label_path = label_dir / raw_path.name
        if label_path.exists():
            pairs.append((raw_path, label_path))
        else:
            missing.append(label_path)
    if missing:
        missing_names = ", ".join(path.name for path in missing[:5])
        raise FileNotFoundError(f"Missing label files: {missing_names}")
    return pairs


def load_seedvig_sessions(
    data_root: str | Path | None = None,
    *,
    raw_data_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    sample_rate: int = 200,
    window_seconds: int = 8,
    label_mode: LabelMode = "threshold35",
    bandpass: bool = False,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
) -> list[SeedVigSession]:
    return load_seedvig_file_pairs(
        discover_seedvig_files(data_root, raw_data_dir=raw_data_dir, label_dir=label_dir),
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        label_mode=label_mode,
        bandpass=bandpass,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
    )


def load_seedvig_file_pairs(
    file_pairs: Iterable[tuple[Path, Path]],
    *,
    sample_rate: int = 200,
    window_seconds: int = 8,
    label_mode: LabelMode = "threshold35",
    bandpass: bool = False,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
) -> list[SeedVigSession]:
    sessions: list[SeedVigSession] = []
    for raw_path, label_path in file_pairs:
        sessions.append(
            load_seedvig_session(
                raw_path,
                label_path,
                sample_rate=sample_rate,
                window_seconds=window_seconds,
                label_mode=label_mode,
                bandpass=bandpass,
                bandpass_low=bandpass_low,
                bandpass_high=bandpass_high,
            )
        )
    return sessions


def load_seedvig_session(
    raw_path: str | Path,
    label_path: str | Path,
    *,
    sample_rate: int = 200,
    window_seconds: int = 8,
    label_mode: LabelMode = "threshold35",
    bandpass: bool = False,
    bandpass_low: float = 1.0,
    bandpass_high: float = 50.0,
) -> SeedVigSession:
    raw_path = Path(raw_path)
    label_path = Path(label_path)
    raw_mat = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
    label_mat = sio.loadmat(label_path, squeeze_me=True, struct_as_record=False)

    eeg_obj = _extract_eeg_object(raw_mat)
    eeg = _extract_eeg_data(eeg_obj).astype(np.float32, copy=False)
    detected_sample_rate = _extract_sample_rate(raw_mat, eeg_obj)
    if detected_sample_rate != sample_rate:
        raise ValueError(
            f"{raw_path.name}: expected sample_rate={sample_rate}, "
            f"found {detected_sample_rate}"
        )

    eeg = _ensure_time_by_channels(eeg, raw_path)
    if eeg.shape[1] != 17:
        raise ValueError(f"{raw_path.name}: expected 17 EEG channels, found {eeg.shape[1]}")

    nan_count = int(np.isnan(eeg).sum())
    inf_count = int(np.isinf(eeg).sum())
    if nan_count or inf_count:
        raise ValueError(f"{raw_path.name}: raw EEG contains {nan_count} NaN and {inf_count} Inf values")

    if bandpass:
        eeg = bandpass_filter(eeg, sample_rate, bandpass_low, bandpass_high)

    perclos = _extract_labels(label_mat).astype(np.float32, copy=False).reshape(-1)
    window_samples = sample_rate * window_seconds
    raw_segment_count = eeg.shape[0] // window_samples
    if raw_segment_count <= 0:
        raise ValueError(f"{raw_path.name}: no complete {window_seconds}s windows found")

    eeg = eeg[: raw_segment_count * window_samples]
    segments = eeg.reshape(raw_segment_count, window_samples, 17).transpose(0, 2, 1)
    if raw_segment_count != 885:
        raise ValueError(f"{raw_path.name}: expected 885 EEG windows, found {raw_segment_count}")
    if len(perclos) != 885:
        raise ValueError(f"{label_path.name}: expected 885 PERCLOS labels, found {len(perclos)}")
    if raw_segment_count != len(perclos):
        raise ValueError(
            f"{raw_path.name}: segment/label mismatch, "
            f"{raw_segment_count} segments vs {len(perclos)} labels"
        )

    label_result = binarize_perclos(perclos, label_mode=label_mode)
    y = label_result.labels
    keep_mask = label_result.valid_mask
    if int(keep_mask.sum()) != len(y):
        raise ValueError(f"{raw_path.name}: usable labels do not match valid sample mask")

    window_ids = np.arange(raw_segment_count, dtype=np.int64)[keep_mask]
    segments = np.ascontiguousarray(segments[keep_mask], dtype=np.float32)
    perclos_kept = np.ascontiguousarray(perclos[keep_mask], dtype=np.float32)

    subject_id = parse_subject_id(raw_path)
    session_id = raw_path.stem
    all_window_ids = np.arange(raw_segment_count, dtype=np.int64)
    all_labels = np.full(raw_segment_count, -1, dtype=np.int64)
    all_labels[keep_mask] = y
    sample_index = pd.DataFrame(
        {
            "sample_id": [f"{session_id}_w{window_id:04d}" for window_id in all_window_ids],
            "subject_id": subject_id,
            "session_id": session_id,
            "file_name": raw_path.name,
            "window_id": all_window_ids,
            "perclos_value": perclos,
            "label": all_labels,
            "label_mode": label_mode,
            "is_valid_binary_sample": keep_mask,
        }
    )
    sample_ids = np.array(
        [f"{session_id}_w{window_id:04d}" for window_id in window_ids],
        dtype=object,
    )
    file_names = np.full(len(window_ids), raw_path.name, dtype=object)

    return SeedVigSession(
        subject_id=subject_id,
        session_id=session_id,
        raw_path=raw_path,
        label_path=label_path,
        label_mode=label_mode,
        x=segments,
        y=y.astype(np.int64, copy=False),
        perclos=perclos_kept,
        window_ids=window_ids,
        sample_ids=sample_ids,
        file_names=file_names,
        sample_index=sample_index,
        raw_segment_count=raw_segment_count,
        dropped_middle_count=label_result.excluded_count,
        alert_count=label_result.alert_count,
        fatigue_count=label_result.fatigue_count,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def binarize_perclos(
    perclos: np.ndarray,
    *,
    label_mode: LabelMode,
) -> LabelResult:
    perclos = np.asarray(perclos).reshape(-1)
    if label_mode == "threshold35":
        valid_mask = np.ones_like(perclos, dtype=bool)
        labels = np.where(perclos <= 0.35, 0, 1).astype(np.int64)
        rule = "PERCLOS <= 0.35 => alert(0); PERCLOS > 0.35 => fatigue(1); no samples discarded"
    elif label_mode == "strict035070":
        alert = perclos < 0.35
        fatigue = perclos > 0.70
        valid_mask = alert | fatigue
        labels = np.where(fatigue[valid_mask], 1, 0).astype(np.int64)
        rule = "PERCLOS < 0.35 => alert(0); PERCLOS > 0.70 => fatigue(1); 0.35 <= PERCLOS <= 0.70 discarded"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    return LabelResult(
        labels=labels,
        valid_mask=valid_mask,
        excluded_count=int((~valid_mask).sum()),
        alert_count=int((labels == 0).sum()),
        fatigue_count=int((labels == 1).sum()),
        label_rule=rule,
    )


def bandpass_filter(
    eeg: np.ndarray,
    sample_rate: int,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if not 0 < low < high < nyquist:
        raise ValueError(f"Invalid bandpass range {low}-{high} Hz for fs={sample_rate}")
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, eeg, axis=0).astype(np.float32, copy=False)


def sessions_to_arrays(sessions: Iterable[SeedVigSession]) -> dict[str, np.ndarray]:
    sessions = list(sessions)
    if not sessions:
        raise ValueError("No sessions provided")
    return {
        "x": np.concatenate([s.x for s in sessions], axis=0),
        "y": np.concatenate([s.y for s in sessions], axis=0),
        "subject_id": np.concatenate(
            [np.full(len(s.y), s.subject_id, dtype=np.int64) for s in sessions]
        ),
        "session_id": np.concatenate(
            [np.full(len(s.y), s.session_id, dtype=object) for s in sessions]
        ),
        "file_name": np.concatenate([s.file_names for s in sessions]),
        "window_id": np.concatenate([s.window_ids for s in sessions]),
        "sample_id": np.concatenate([s.sample_ids for s in sessions]),
        "perclos_value": np.concatenate([s.perclos for s in sessions]),
        "label_mode": np.concatenate(
            [np.full(len(s.y), s.label_mode, dtype=object) for s in sessions]
        ),
        "is_valid_binary_sample": np.concatenate(
            [np.full(len(s.y), True, dtype=bool) for s in sessions]
        ),
    }


def compute_channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=False).astype(np.float32).reshape(17, 1)
    std = x.std(axis=(0, 2), keepdims=False).astype(np.float32).reshape(17, 1)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_channel_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 17, 1)) / std.reshape(1, 17, 1)).astype(np.float32)


def compute_robust_clip_bounds(
    x: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(x, lower_percentile, axis=(0, 2)).astype(np.float32).reshape(17, 1)
    hi = np.percentile(x, upper_percentile, axis=(0, 2)).astype(np.float32).reshape(17, 1)
    return lo, hi


def apply_robust_clip(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(x, lo.reshape(1, 17, 1), hi.reshape(1, 17, 1)).astype(np.float32)


def nan_inf_counts(x: np.ndarray) -> tuple[int, int]:
    return int(np.isnan(x).sum()), int(np.isinf(x).sum())


def _extract_eeg_object(mat: dict) -> object:
    if "EEG" in mat:
        return mat["EEG"]
    return mat


def _extract_eeg_data(eeg_obj: object) -> np.ndarray:
    for key in ("data", "Data", "eeg", "EEG"):
        value = _get_field(eeg_obj, key)
        if value is not None and isinstance(value, np.ndarray) and value.ndim == 2:
            return value
    if isinstance(eeg_obj, dict):
        numeric = [
            value for key, value in eeg_obj.items()
            if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2
        ]
        if len(numeric) == 1:
            return numeric[0]
    raise KeyError("Could not find 2D EEG data array in .mat file")


def _extract_sample_rate(mat: dict, eeg_obj: object) -> int:
    for container in (eeg_obj, mat):
        for key in ("sample_rate", "sampling_rate", "srate", "fs", "fsample"):
            value = _get_field(container, key)
            if value is not None:
                arr = np.asarray(value).reshape(-1)
                if arr.size:
                    return int(round(float(arr[0])))
    raise KeyError("Could not find sample rate field in raw .mat file")


def _extract_labels(mat: dict) -> np.ndarray:
    preferred = ("perclos", "PERCLOS", "perclos_labels", "labels", "label", "y")
    for key in preferred:
        value = _get_field(mat, key)
        if value is not None:
            arr = np.asarray(value)
            if arr.size:
                return arr.reshape(-1)

    numeric = [
        np.asarray(value).reshape(-1)
        for key, value in mat.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and np.asarray(value).size
    ]
    numeric = [arr for arr in numeric if np.issubdtype(arr.dtype, np.number)]
    if len(numeric) == 1:
        return numeric[0]
    raise KeyError("Could not find PERCLOS label array in .mat file")


def _get_field(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, np.ndarray) and obj.dtype.names and key in obj.dtype.names:
        return obj[key]
    return None


def _ensure_time_by_channels(eeg: np.ndarray, raw_path: Path) -> np.ndarray:
    if eeg.ndim != 2:
        raise ValueError(f"{raw_path.name}: expected 2D EEG array, found shape {eeg.shape}")
    if eeg.shape[1] == 17:
        return eeg
    if eeg.shape[0] == 17:
        return eeg.T
    raise ValueError(f"{raw_path.name}: cannot infer 17-channel EEG orientation from {eeg.shape}")
