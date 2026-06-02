from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eegda.datasets.standard_npz import save_standard_dataset


EVENT_LEFT_DEVIATION = 251
EVENT_RIGHT_DEVIATION = 252
EVENT_RESPONSE_ONSET = 253
EVENT_RESPONSE_OFFSET = 254
EVENT_CODES = (EVENT_LEFT_DEVIATION, EVENT_RIGHT_DEVIATION, EVENT_RESPONSE_ONSET, EVENT_RESPONSE_OFFSET)
EXPECTED_ORIGINAL_SFREQ = 500.0
EXPECTED_EEG_CHANNELS = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FT7",
    "FC3",
    "FCZ",
    "FC4",
    "FT8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "OZ",
    "O2",
)


@dataclass(frozen=True)
class SetSource:
    set_path: Path
    fdt_path: Path | None
    zip_path: Path | None
    set_member: str | None
    fdt_member: str | None
    session_id: str


@dataclass
class TrialRecord:
    subject_id: int
    session_id: str
    trial_id: int
    deviation_code: int
    deviation_sample: int | None = None
    response_sample: int | None = None
    response_offset_sample: int | None = None
    deviation_time_sec: float = np.nan
    response_time_sec: float = np.nan
    local_rt: float = np.nan
    global_rt: float = np.nan
    alert_rt: float = np.nan
    label_state: str = "invalid"
    included_in_dataset: bool = False
    invalid_reason: str = ""


@dataclass
class SessionAnalysis:
    source: SetSource
    subject_id: int
    sfreq: float = np.nan
    n_channels: int = 0
    original_n_channels: int = 0
    channel_names: list[str] | None = None
    kept_channel_indices: list[int] | None = None
    dropped_channel_names: list[str] | None = None
    n_points: int = 0
    duration_sec: float = np.nan
    event_counts: dict[int, int] | None = None
    unknown_event_count: int = 0
    trials: list[TrialRecord] | None = None
    candidate_deviation_count: int = 0
    valid_rt_count: int = 0
    invalid_rt_count: int = 0
    epoch_available_count: int = 0
    no_global_count: int = 0
    alert_count: int = 0
    fatigue_count: int = 0
    transition_count: int = 0
    included_before_subject_selection: bool = False
    selected_for_subject: bool = False
    exclusion_reason: str = ""
    alert_rt: float = np.nan
    local_rt_min: float = np.nan
    local_rt_max: float = np.nan
    local_rt_mean: float = np.nan
    local_rt_median: float = np.nan
    nan_count: int | None = None
    inf_count: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SADT RT-labelled unbalanced EEGDA standard cache.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sfreq-out", type=int, default=128)
    parser.add_argument("--epoch-seconds", type=float, default=3.0)
    parser.add_argument("--rt-cleaning", choices=("none", "range"), default="none")
    parser.add_argument("--rt-min-sec", type=float, default=0.30)
    parser.add_argument("--rt-max-sec", type=float, default=10.0)
    parser.add_argument("--global-rt-window-sec", type=float, default=90.0)
    parser.add_argument(
        "--global-rt-mode",
        choices=("previous_only", "previous_window", "include_current_window"),
        default="previous_only",
        help="How to compute global RT. previous_window is kept as a backward-compatible alias for previous_only.",
    )
    parser.add_argument("--min-samples-per-class", type=int, default=50)
    parser.add_argument("--session-policy", choices=("one_most_balanced", "all_valid"), default="one_most_balanced")
    parser.add_argument(
        "--subject-session-selection",
        choices=("most_balanced", "largest_total", "largest_minority"),
        default="most_balanced",
        help="Selection rule when one session per subject is kept.",
    )
    parser.add_argument("--balance-mode", choices=("none",), default="none")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    if args.rt_cleaning == "range" and (args.rt_min_sec < 0 or args.rt_max_sec <= args.rt_min_sec):
        raise ValueError("--rt-max-sec must be greater than --rt-min-sec")
    if args.sfreq_out <= 0:
        raise ValueError("--sfreq-out must be positive")

    sources = discover_set_sources(Path(args.input_dir))
    if args.max_sessions is not None:
        sources = sources[: args.max_sessions]
    if not sources:
        raise FileNotFoundError(f"No .set or .set.zip files found under {args.input_dir}")

    analyses = [analyze_session(source, args) for source in sources]
    select_sessions(analyses, args)
    apply_final_trial_inclusion_flags(analyses)
    selected = [analysis for analysis in analyses if analysis.selected_for_subject]
    if not selected:
        raise RuntimeError("No SADT sessions passed filtering and subject selection.")

    X, y, subjects, sessions, sample_ids, extras, channel_names = extract_selected_epochs(selected, args)
    metadata = build_metadata(args, selected, analyses)
    save_standard_dataset(
        output_path,
        X=X,
        y=y,
        subjects=subjects,
        sessions=sessions,
        sample_ids=sample_ids,
        sfreq=args.sfreq_out,
        channel_names=channel_names,
        label_names={0: "alert", 1: "fatigue"},
        metadata=metadata,
        extra_arrays=extras,
    )
    write_reports(output_path, analyses)
    print_summary(args, output_path, analyses, selected, X, y)


def discover_set_sources(input_dir: Path) -> list[SetSource]:
    sources: list[SetSource] = []
    for set_path in sorted(input_dir.rglob("*.set")):
        fdt_path = set_path.with_suffix(".fdt")
        session_id = set_path.stem
        sources.append(
            SetSource(
                set_path=set_path,
                fdt_path=fdt_path if fdt_path.exists() else None,
                zip_path=None,
                set_member=None,
                fdt_member=None,
                session_id=session_id,
            )
        )
    seen_sessions = {source.session_id for source in sources}
    for zip_path in sorted(input_dir.rglob("*.set.zip")):
        with ZipFile(zip_path) as zf:
            set_members = [name for name in zf.namelist() if name.endswith(".set")]
            fdt_members = [name for name in zf.namelist() if name.endswith(".fdt")]
        if not set_members:
            continue
        set_member = set_members[0]
        session_id = Path(set_member).stem
        if session_id in seen_sessions:
            continue
        fdt_member = next((name for name in fdt_members if Path(name).stem == session_id), None)
        sources.append(
            SetSource(
                set_path=zip_path,
                fdt_path=None,
                zip_path=zip_path,
                set_member=set_member,
                fdt_member=fdt_member,
                session_id=session_id,
            )
        )
    return sorted(sources, key=lambda source: source.session_id)


def analyze_session(source: SetSource, args: argparse.Namespace) -> SessionAnalysis:
    subject_id = parse_subject_id(source.session_id)
    analysis = SessionAnalysis(source=source, subject_id=subject_id, event_counts={code: 0 for code in EVENT_CODES}, trials=[])
    try:
        eeg = load_eeglab_metadata(source)
        analysis.sfreq = float(eeg.srate)
        analysis.original_n_channels = int(eeg.nbchan)
        analysis.n_channels = analysis.original_n_channels
        analysis.n_points = int(eeg.pnts)
        analysis.duration_sec = analysis.n_points / analysis.sfreq
        analysis.channel_names = extract_channel_names(eeg)
        if source.zip_path is None and source.fdt_path is None:
            analysis.exclusion_reason = "missing_fdt_file"
            return analysis
        if source.zip_path is not None and source.fdt_member is None:
            analysis.exclusion_reason = "missing_fdt_in_zip"
            return analysis
        if analysis.n_channels < 30:
            analysis.exclusion_reason = "fewer_than_30_channels"
            return analysis
        if analysis.n_channels > 30:
            indices = resolve_eeg_channel_indices(analysis.channel_names)
            if indices is None:
                analysis.exclusion_reason = "more_than_30_channels_without_clear_eeg_mapping"
                return analysis
            analysis.kept_channel_indices = indices
            kept = set(indices)
            analysis.dropped_channel_names = [
                name for idx, name in enumerate(analysis.channel_names or []) if idx not in kept
            ]
            analysis.channel_names = [analysis.channel_names[idx] for idx in indices]
            analysis.n_channels = len(indices)
        else:
            analysis.kept_channel_indices = list(range(analysis.n_channels))
        if int(round(analysis.sfreq)) != int(EXPECTED_ORIGINAL_SFREQ):
            analysis.exclusion_reason = f"unexpected_sfreq_{analysis.sfreq:g}"
            return analysis
        events, unknown_event_count = extract_events(eeg)
        analysis.unknown_event_count = unknown_event_count
        counts = {code: 0 for code in EVENT_CODES}
        for event in events:
            if event["code"] in counts:
                counts[event["code"]] += 1
        analysis.event_counts = counts
        analysis.candidate_deviation_count = counts[EVENT_LEFT_DEVIATION] + counts[EVENT_RIGHT_DEVIATION]
        analysis.trials = build_trials(events, analysis, args)
        label_trials(analysis, args)
    except Exception as exc:  # noqa: BLE001 - report and continue to next session
        analysis.exclusion_reason = f"read_error:{type(exc).__name__}:{exc}"
    return analysis


def load_eeglab_metadata(source: SetSource):
    if source.zip_path is not None:
        assert source.set_member is not None
        with ZipFile(source.zip_path) as zf:
            raw = zf.read(source.set_member)
        return sio.loadmat(BytesIO(raw), squeeze_me=True, struct_as_record=False)["EEG"]
    return sio.loadmat(source.set_path, squeeze_me=True, struct_as_record=False)["EEG"]


def extract_channel_names(eeg) -> list[str]:
    names = []
    for chan in np.atleast_1d(eeg.chanlocs):
        names.append(str(getattr(chan, "labels", "")))
    return names


def normalize_channel_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def resolve_eeg_channel_indices(channel_names: list[str] | None) -> list[int] | None:
    if not channel_names:
        return None
    normalized_to_index = {normalize_channel_name(name): idx for idx, name in enumerate(channel_names)}
    indices = []
    for expected in EXPECTED_EEG_CHANNELS:
        idx = normalized_to_index.get(normalize_channel_name(expected))
        if idx is None:
            return None
        indices.append(idx)
    if len(set(indices)) != len(EXPECTED_EEG_CHANNELS):
        return None
    return indices


def parse_event_code(raw_code: object) -> int | None:
    if isinstance(raw_code, bytes):
        raw_code = raw_code.decode(errors="ignore")
    if isinstance(raw_code, str):
        text = raw_code.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if not numeric.is_integer():
            return None
        return int(numeric)
    try:
        numeric = float(raw_code)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def extract_events(eeg) -> tuple[list[dict[str, float | int]], int]:
    events = []
    unknown_event_count = 0
    for raw_event in np.atleast_1d(eeg.event):
        code = parse_event_code(getattr(raw_event, "type"))
        if code is None:
            unknown_event_count += 1
            continue
        latency = int(round(float(getattr(raw_event, "latency"))))
        if code not in EVENT_CODES:
            unknown_event_count += 1
        events.append({"code": code, "latency": latency})
    return sorted(events, key=lambda event: int(event["latency"])), unknown_event_count


def build_trials(events: list[dict[str, float | int]], analysis: SessionAnalysis, args: argparse.Namespace) -> list[TrialRecord]:
    trials: list[TrialRecord] = []
    deviation_indices = [idx for idx, event in enumerate(events) if event["code"] in (251, 252)]
    for trial_id, event_idx in enumerate(deviation_indices):
        event = events[event_idx]
        deviation_sample = int(event["latency"])
        next_deviation_sample = None
        if trial_id + 1 < len(deviation_indices):
            next_deviation_sample = int(events[deviation_indices[trial_id + 1]]["latency"])
        trial = TrialRecord(
            subject_id=analysis.subject_id,
            session_id=analysis.source.session_id,
            trial_id=trial_id,
            deviation_code=int(event["code"]),
            deviation_sample=deviation_sample,
            deviation_time_sec=deviation_sample / analysis.sfreq,
        )
        response = next(
            (
                candidate
                for candidate in events[event_idx + 1 :]
                if candidate["code"] == EVENT_RESPONSE_ONSET
                and (next_deviation_sample is None or int(candidate["latency"]) < next_deviation_sample)
            ),
            None,
        )
        if response is None:
            trial.invalid_reason = "missing_response_onset_before_next_deviation"
            trials.append(trial)
            continue
        trial.response_sample = int(response["latency"])
        trial.response_time_sec = trial.response_sample / analysis.sfreq
        offset = next(
            (
                candidate
                for candidate in events[event_idx + 1 :]
                if candidate["code"] == EVENT_RESPONSE_OFFSET and int(candidate["latency"]) > trial.response_sample
            ),
            None,
        )
        if offset is not None:
            trial.response_offset_sample = int(offset["latency"])
        trial.local_rt = (trial.response_sample - deviation_sample) / analysis.sfreq
        epoch_start = deviation_sample - int(round(args.epoch_seconds * analysis.sfreq))
        epoch_end = deviation_sample
        if trial.response_sample <= deviation_sample:
            trial.invalid_reason = "response_onset_not_after_deviation"
        elif args.rt_cleaning == "range" and trial.local_rt <= args.rt_min_sec:
            trial.invalid_reason = "rt_too_short"
        elif args.rt_cleaning == "range" and trial.local_rt > args.rt_max_sec:
            trial.invalid_reason = "rt_too_long"
        elif epoch_start < 0:
            trial.invalid_reason = "insufficient_pre_event_eeg"
        elif epoch_end > analysis.n_points:
            trial.invalid_reason = "epoch_out_of_bounds"
        else:
            trial.label_state = "valid_unlabeled"
        trials.append(trial)
    return trials


def label_trials(analysis: SessionAnalysis, args: argparse.Namespace) -> None:
    assert analysis.trials is not None
    valid_trials = [trial for trial in analysis.trials if trial.label_state == "valid_unlabeled"]
    analysis.valid_rt_count = len(valid_trials)
    analysis.invalid_rt_count = len(analysis.trials) - len(valid_trials)
    analysis.epoch_available_count = len(valid_trials)
    if not valid_trials:
        analysis.exclusion_reason = "no_valid_rt_trials"
        return
    valid_rts = np.asarray([trial.local_rt for trial in valid_trials], dtype=np.float64)
    analysis.alert_rt = float(np.percentile(valid_rts, 5))
    analysis.local_rt_min = float(np.min(valid_rts))
    analysis.local_rt_max = float(np.max(valid_rts))
    analysis.local_rt_mean = float(np.mean(valid_rts))
    analysis.local_rt_median = float(np.median(valid_rts))
    ordered_valid = sorted(valid_trials, key=lambda item: item.deviation_time_sec)
    previous_valid: list[TrialRecord] = []
    for trial in ordered_valid:
        trial.alert_rt = analysis.alert_rt
        prior_candidates = previous_valid
        if args.global_rt_mode == "include_current_window":
            prior_candidates = previous_valid + [trial]
        prior = [
            item.local_rt
            for item in prior_candidates
            if 0 <= trial.deviation_time_sec - item.deviation_time_sec <= args.global_rt_window_sec
        ]
        if not prior:
            trial.label_state = "no_global"
            trial.invalid_reason = "no_previous_valid_rt_in_global_window"
            analysis.no_global_count += 1
        else:
            trial.global_rt = float(np.mean(prior))
            if trial.local_rt < 1.5 * analysis.alert_rt and trial.global_rt < 1.5 * analysis.alert_rt:
                trial.label_state = "alert"
                trial.included_in_dataset = True
                analysis.alert_count += 1
            elif trial.local_rt > 2.5 * analysis.alert_rt and trial.global_rt > 2.5 * analysis.alert_rt:
                trial.label_state = "fatigue"
                trial.included_in_dataset = True
                analysis.fatigue_count += 1
            else:
                trial.label_state = "transition"
                trial.invalid_reason = "transition_rt_state"
                analysis.transition_count += 1
        previous_valid.append(trial)
    exclusion_reasons = []
    if analysis.alert_count < args.min_samples_per_class:
        exclusion_reasons.append(f"alert_count<{args.min_samples_per_class}")
    if analysis.fatigue_count < args.min_samples_per_class:
        exclusion_reasons.append(f"fatigue_count<{args.min_samples_per_class}")
    if exclusion_reasons:
        analysis.exclusion_reason = ";".join(exclusion_reasons)
    else:
        analysis.included_before_subject_selection = True


def select_sessions(analyses: list[SessionAnalysis], args: argparse.Namespace) -> None:
    valid = [analysis for analysis in analyses if analysis.included_before_subject_selection]
    if args.session_policy == "all_valid":
        for analysis in valid:
            analysis.selected_for_subject = True
        return
    by_subject: dict[int, list[SessionAnalysis]] = {}
    for analysis in valid:
        by_subject.setdefault(analysis.subject_id, []).append(analysis)
    for subject_sessions in by_subject.values():
        selected = sorted(subject_sessions, key=lambda item: session_selection_key(item, args))[0]
        selected.selected_for_subject = True
        for analysis in subject_sessions:
            if analysis is not selected:
                analysis.exclusion_reason = "not_selected_by_session_policy"


def apply_final_trial_inclusion_flags(analyses: list[SessionAnalysis]) -> None:
    for analysis in analyses:
        if analysis.selected_for_subject:
            continue
        reason = analysis.exclusion_reason or "session_not_selected"
        for trial in analysis.trials or []:
            if trial.included_in_dataset:
                trial.included_in_dataset = False
                trial.invalid_reason = reason


def session_selection_key(analysis: SessionAnalysis, args: argparse.Namespace) -> tuple[int, int, int, str]:
    class_imbalance = abs(analysis.alert_count - analysis.fatigue_count)
    minority_count = min(analysis.alert_count, analysis.fatigue_count)
    total_count = analysis.alert_count + analysis.fatigue_count
    if args.subject_session_selection == "most_balanced":
        return (class_imbalance, -minority_count, -total_count, analysis.source.session_id)
    if args.subject_session_selection == "largest_total":
        return (-total_count, class_imbalance, -minority_count, analysis.source.session_id)
    if args.subject_session_selection == "largest_minority":
        return (-minority_count, class_imbalance, -total_count, analysis.source.session_id)
    raise ValueError(f"Unsupported subject_session_selection: {args.subject_session_selection}")


def extract_selected_epochs(
    selected: list[SessionAnalysis],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    X_parts = []
    y_parts = []
    subject_parts = []
    session_parts = []
    sample_id_parts = []
    extras: dict[str, list[np.ndarray]] = {
        "local_rt": [],
        "global_rt": [],
        "alert_rt": [],
        "deviation_code": [],
        "deviation_time_sec": [],
        "response_time_sec": [],
    }
    channel_names = selected[0].channel_names or [f"EEG{i + 1}" for i in range(30)]
    for analysis in selected:
        included_trials = [trial for trial in analysis.trials or [] if trial.included_in_dataset]
        if not included_trials:
            continue
        raw = load_session_eeg_data(analysis)
        nan_count = int(np.isnan(raw).sum())
        inf_count = int(np.isinf(raw).sum())
        analysis.nan_count = nan_count
        analysis.inf_count = inf_count
        if nan_count or inf_count:
            analysis.selected_for_subject = False
            analysis.exclusion_reason = f"raw_eeg_nan_inf:{nan_count}:{inf_count}"
            continue
        epochs = []
        labels = []
        sample_ids = []
        local_rt = []
        global_rt = []
        alert_rt = []
        deviation_code = []
        deviation_time_sec = []
        response_time_sec = []
        for trial in included_trials:
            epoch = extract_epoch(raw, trial, analysis.sfreq, args)
            if epoch is None:
                trial.included_in_dataset = False
                trial.invalid_reason = "epoch_resample_shape_mismatch"
                continue
            epochs.append(epoch)
            labels.append(0 if trial.label_state == "alert" else 1)
            sample_ids.append(f"{analysis.source.session_id}_trial{trial.trial_id:04d}")
            local_rt.append(trial.local_rt)
            global_rt.append(trial.global_rt)
            alert_rt.append(trial.alert_rt)
            deviation_code.append(trial.deviation_code)
            deviation_time_sec.append(trial.deviation_time_sec)
            response_time_sec.append(trial.response_time_sec)
        if not epochs:
            continue
        n = len(epochs)
        X_parts.append(np.stack(epochs, axis=0).astype(np.float32, copy=False))
        y_parts.append(np.asarray(labels, dtype=np.int64))
        subject_parts.append(np.full(n, analysis.subject_id, dtype=np.int64))
        session_parts.append(np.full(n, analysis.source.session_id, dtype=object))
        sample_id_parts.append(np.asarray(sample_ids, dtype=object))
        extras["local_rt"].append(np.asarray(local_rt, dtype=np.float32))
        extras["global_rt"].append(np.asarray(global_rt, dtype=np.float32))
        extras["alert_rt"].append(np.asarray(alert_rt, dtype=np.float32))
        extras["deviation_code"].append(np.asarray(deviation_code, dtype=np.int64))
        extras["deviation_time_sec"].append(np.asarray(deviation_time_sec, dtype=np.float32))
        extras["response_time_sec"].append(np.asarray(response_time_sec, dtype=np.float32))
    if not X_parts:
        raise RuntimeError("Selected sessions did not produce any epochs.")
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
        np.concatenate(session_parts, axis=0),
        np.concatenate(sample_id_parts, axis=0),
        {key: np.concatenate(values, axis=0) for key, values in extras.items()},
        channel_names,
    )


def load_session_eeg_data(analysis: SessionAnalysis) -> np.ndarray:
    source = analysis.source
    source_n_channels = analysis.original_n_channels or analysis.n_channels
    n_values = source_n_channels * analysis.n_points
    if source.zip_path is not None:
        assert source.fdt_member is not None
        with ZipFile(source.zip_path) as zf:
            raw_bytes = zf.read(source.fdt_member)
        values = np.frombuffer(raw_bytes, dtype="<f4", count=n_values)
    else:
        assert source.fdt_path is not None
        values = np.memmap(source.fdt_path, dtype="<f4", mode="r", shape=(n_values,))
    if values.size != n_values:
        raise ValueError(f"{source.session_id}: expected {n_values} fdt values, found {values.size}")
    raw = np.asarray(values).reshape(analysis.n_points, source_n_channels).T
    if analysis.kept_channel_indices is not None:
        raw = raw[np.asarray(analysis.kept_channel_indices, dtype=np.int64)]
    return raw


def extract_epoch(raw: np.ndarray, trial: TrialRecord, sfreq: float, args: argparse.Namespace) -> np.ndarray | None:
    assert trial.deviation_sample is not None
    input_samples = int(round(args.epoch_seconds * sfreq))
    start = trial.deviation_sample - input_samples
    stop = trial.deviation_sample
    if start < 0 or stop > raw.shape[1]:
        return None
    epoch = raw[:, start:stop]
    up, down = rational_resample_factors(args.sfreq_out, sfreq)
    epoch = signal.resample_poly(epoch, up, down, axis=1).astype(np.float32, copy=False)
    expected_samples = int(round(args.epoch_seconds * args.sfreq_out))
    if epoch.shape != (30, expected_samples):
        return None
    return epoch


def rational_resample_factors(sfreq_out: float, sfreq_in: float) -> tuple[int, int]:
    from fractions import Fraction

    fraction = Fraction(float(sfreq_out) / float(sfreq_in)).limit_denominator(1000)
    return fraction.numerator, fraction.denominator


def build_metadata(args: argparse.Namespace, selected: list[SessionAnalysis], analyses: list[SessionAnalysis]) -> dict[str, object]:
    return {
        "dataset_name": "sadt",
        "protocol_name": infer_protocol_name(args),
        "source_format": "official preprocessed continuous EEGLAB .set/.fdt",
        "input_channels": 30,
        "input_samples": int(round(args.epoch_seconds * args.sfreq_out)),
        "sfreq_original": int(EXPECTED_ORIGINAL_SFREQ),
        "sfreq_out": args.sfreq_out,
        "epoch_seconds": args.epoch_seconds,
        "rt_cleaning": args.rt_cleaning,
        "rt_min_sec": args.rt_min_sec,
        "rt_max_sec": args.rt_max_sec,
        "global_rt_window_sec": args.global_rt_window_sec,
        "global_rt_mode": args.global_rt_mode,
        "min_samples_per_class": args.min_samples_per_class,
        "session_policy": args.session_policy,
        "subject_session_selection": args.subject_session_selection,
        "balance_mode": args.balance_mode,
        "transition_handling": "discarded",
        "normalization": "not_applied_in_cache",
        "robust_clipping": "not_applied_in_cache",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processed_session_count": len(analyses),
        "selected_session_count": len(selected),
        "selected_subject_count": len({analysis.subject_id for analysis in selected}),
    }


def infer_protocol_name(args: argparse.Namespace) -> str:
    output_name = Path(args.output_path).stem if hasattr(args, "output_path") else ""
    if "icnn_text" in output_name:
        return "sadt-rt-icnn-text-unbalanced"
    if "icnn_compatible" in output_name:
        return "sadt-rt-icnn-compatible-unbalanced"
    return "sadt-rt-unbalanced"


def write_reports(output_path: Path, analyses: list[SessionAnalysis]) -> None:
    prefix = output_path.with_suffix("")
    session_rows = [session_report_row(analysis) for analysis in analyses]
    trial_rows = [trial_report_row(trial) for analysis in analyses for trial in (analysis.trials or [])]
    subject_rows = subject_report_rows(analyses)
    pd.DataFrame(session_rows).to_csv(prefix.with_name(f"{prefix.name}_session_report.csv"), index=False)
    pd.DataFrame(trial_rows).to_csv(prefix.with_name(f"{prefix.name}_trial_report.csv"), index=False)
    pd.DataFrame(subject_rows).to_csv(prefix.with_name(f"{prefix.name}_subject_report.csv"), index=False)


def session_report_row(analysis: SessionAnalysis) -> dict[str, object]:
    counts = analysis.event_counts or {}
    return {
        "subject_id": analysis.subject_id,
        "session_id": analysis.source.session_id,
        "set_path": str(analysis.source.set_path),
        "sfreq": analysis.sfreq,
        "n_channels": analysis.n_channels,
        "original_n_channels": analysis.original_n_channels,
        "dropped_channel_names": "" if not analysis.dropped_channel_names else "|".join(analysis.dropped_channel_names),
        "duration_sec": analysis.duration_sec,
        "event_251_count": counts.get(251, 0),
        "event_252_count": counts.get(252, 0),
        "event_253_count": counts.get(253, 0),
        "event_254_count": counts.get(254, 0),
        "unknown_event_count": analysis.unknown_event_count,
        "candidate_deviation_count": analysis.candidate_deviation_count,
        "valid_pair_count": analysis.valid_rt_count,
        "invalid_pair_count": analysis.invalid_rt_count,
        "epoch_available_count": analysis.epoch_available_count,
        "no_global_count": analysis.no_global_count,
        "alert_count": analysis.alert_count,
        "fatigue_count": analysis.fatigue_count,
        "transition_count": analysis.transition_count,
        "included_before_subject_selection": analysis.included_before_subject_selection,
        "selected_for_subject": analysis.selected_for_subject,
        "exclusion_reason": analysis.exclusion_reason,
        "alert_rt": analysis.alert_rt,
        "local_rt_min": analysis.local_rt_min,
        "local_rt_5th": analysis.alert_rt,
        "local_rt_max": analysis.local_rt_max,
        "local_rt_mean": analysis.local_rt_mean,
        "local_rt_median": analysis.local_rt_median,
    }


def trial_report_row(trial: TrialRecord) -> dict[str, object]:
    return {
        "subject_id": trial.subject_id,
        "session_id": trial.session_id,
        "trial_id": trial.trial_id,
        "deviation_code": trial.deviation_code,
        "deviation_time_sec": trial.deviation_time_sec,
        "response_time_sec": trial.response_time_sec,
        "local_rt": trial.local_rt,
        "global_rt": trial.global_rt,
        "alert_rt": trial.alert_rt,
        "label_state": trial.label_state,
        "included_in_dataset": trial.included_in_dataset,
        "invalid_reason": trial.invalid_reason,
    }


def subject_report_rows(analyses: list[SessionAnalysis]) -> list[dict[str, object]]:
    rows = []
    for subject_id in sorted({analysis.subject_id for analysis in analyses}):
        subject_sessions = [analysis for analysis in analyses if analysis.subject_id == subject_id]
        selected = next((analysis for analysis in subject_sessions if analysis.selected_for_subject), None)
        rows.append(
            {
                "subject_id": subject_id,
                "selected_session_id": "" if selected is None else selected.source.session_id,
                "alert_count": 0 if selected is None else selected.alert_count,
                "fatigue_count": 0 if selected is None else selected.fatigue_count,
                "total_binary_samples": 0 if selected is None else selected.alert_count + selected.fatigue_count,
                "class_imbalance": np.nan if selected is None else abs(selected.alert_count - selected.fatigue_count),
                "n_valid_sessions": sum(analysis.included_before_subject_selection for analysis in subject_sessions),
                "n_excluded_sessions": sum(not analysis.included_before_subject_selection for analysis in subject_sessions),
            }
        )
    return rows


def print_summary(
    args: argparse.Namespace,
    output_path: Path,
    analyses: list[SessionAnalysis],
    selected: list[SessionAnalysis],
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    print("sadt_rt_standard_cache_summary")
    print(f"  output_path={output_path}")
    print(f"  sessions_processed={len(analyses)}")
    print(f"  sessions_valid_before_subject_selection={sum(a.included_before_subject_selection for a in analyses)}")
    print(f"  sessions_selected={len(selected)}")
    print(f"  subjects_selected={sorted({a.subject_id for a in selected})}")
    print(f"  sfreq_out={args.sfreq_out}")
    print(f"  epoch_seconds={args.epoch_seconds}")
    print(f"  final_X_shape={X.shape}")
    print(f"  final_y_shape={y.shape}")
    print(f"  alert_count={int((y == 0).sum())}")
    print(f"  fatigue_count={int((y == 1).sum())}")
    print(f"  transition_count={sum(a.transition_count for a in analyses)}")
    print(f"  no_global_count={sum(a.no_global_count for a in analyses)}")
    print(f"  invalid_rt_count={sum(a.invalid_rt_count for a in analyses)}")
    print(f"  final_nan_count={int(np.isnan(X).sum())}")
    print(f"  final_inf_count={int(np.isinf(X).sum())}")
    print(f"  session_policy={args.session_policy}")
    print(f"  rt_cleaning={args.rt_cleaning}")
    print(f"  global_rt_mode={args.global_rt_mode}")
    print(f"  subject_session_selection={args.subject_session_selection}")
    print(f"  balance_mode={args.balance_mode}")
    print("selected_sessions")
    for analysis in selected:
        print(
            f"  subject={analysis.subject_id} session={analysis.source.session_id} "
            f"alert={analysis.alert_count} fatigue={analysis.fatigue_count} "
            f"alert_rt={analysis.alert_rt:.4f}"
        )


def parse_subject_id(session_id: str) -> int:
    match = re.match(r"^s(\d+)_", session_id)
    if not match:
        raise ValueError(f"Cannot parse SADT subject id from session id: {session_id}")
    return int(match.group(1))


if __name__ == "__main__":
    main()
