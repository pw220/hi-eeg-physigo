from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EEGDAResults:
    """Lightweight result handle returned by ``eegda.run``."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.output_dir = metadata.get("output_dir")
        self.artifacts = self._load_artifacts()
        self.fold_metrics = self._load_fold_metrics()
        self.aggregate_metrics = self._load_aggregate_metrics()
        self.best_fold = self._pick_fold(best=True)
        self.worst_fold = self._pick_fold(best=False)

    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def to_dataframe(self):
        return self.fold_metrics

    def summary(self) -> None:
        if not self.output_dir:
            print("EEGDA Results")
            print("No output directory was saved for this run.")
            return
        rows = _records(self.fold_metrics)
        print("Final Selected-LOSO Summary")
        print("-" * 27)
        print(f"Completed selected folds: {len(rows)}")
        if rows:
            print("")
            print("Per-fold results")
            columns = [
                "target_subject",
                "target_subject_raw",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "fatigue_recall",
                "specificity",
                "roc_auc",
                "auprc",
            ]
            print(_format_table(rows, columns))
        if self.aggregate_metrics:
            print("")
            print("Aggregate results")
            print("metric             mean     std      min      max")
            for key in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "fatigue_precision",
                "fatigue_recall",
                "specificity",
                "roc_auc",
                "auprc",
            ]:
                item = self.aggregate_metrics.get(key)
                if item:
                    print(f"{key:<18} {item['mean']:.4f}   {item['std']:.4f}   {item['min']:.4f}   {item['max']:.4f}")
        if self.best_fold is not None:
            print("")
            print(
                f"Best fold  : subject {self.best_fold.get('target_subject')} "
                f"(raw={self.best_fold.get('target_subject_raw')}) | macro_f1={float(self.best_fold.get('macro_f1')):.4f}"
            )
        if self.worst_fold is not None:
            print(
                f"Worst fold : subject {self.worst_fold.get('target_subject')} "
                f"(raw={self.worst_fold.get('target_subject_raw')}) | macro_f1={float(self.worst_fold.get('macro_f1')):.4f}"
            )
        print("")
        print("Artifacts saved to:")
        print(self.output_dir)

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            if not self.output_dir:
                raise ValueError("Cannot save results without an output directory; pass an explicit path.")
            path = Path(self.output_dir) / "results.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": _jsonable(self.metadata),
            "artifacts": _jsonable(self.artifacts),
            "aggregate_metrics": _jsonable(self.aggregate_metrics),
            "best_fold": _jsonable(self.best_fold),
            "worst_fold": _jsonable(self.worst_fold),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _load_artifacts(self) -> dict[str, Any]:
        if not self.output_dir:
            return {}
        path = Path(self.output_dir) / "artifacts.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"run_dir": self.output_dir}

    def _load_fold_metrics(self):
        if not self.output_dir:
            return []
        path = Path(self.output_dir) / "metrics" / "fold_metrics.csv"
        if not path.exists():
            return []
        try:
            import pandas as pd

            return pd.read_csv(path)
        except Exception:  # noqa: BLE001 - pandas is optional for the public API
            import csv

            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

    def _load_aggregate_metrics(self) -> dict[str, Any]:
        if not self.output_dir:
            return {}
        path = Path(self.output_dir) / "metrics" / "aggregate_metrics.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("metrics", {})

    def _pick_fold(self, *, best: bool) -> dict[str, Any] | None:
        rows = _records(self.fold_metrics)
        rows = [row for row in rows if row.get("macro_f1") not in {None, ""}]
        if not rows:
            return None
        return max(rows, key=lambda row: float(row["macro_f1"])) if best else min(rows, key=lambda row: float(row["macro_f1"]))


DrowEEGResults = EEGDAResults


def _records(table: Any) -> list[dict[str, Any]]:
    if hasattr(table, "to_dict"):
        return table.to_dict(orient="records")
    return list(table or [])


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    labels = {
        "target_subject": "subject",
        "target_subject_raw": "raw_id",
        "balanced_accuracy": "balanced_acc",
    }
    headers = [labels.get(column, column) for column in columns]
    formatted_rows = []
    for row in rows:
        formatted = []
        for column in columns:
            value = row.get(column, "")
            if column not in {"target_subject", "target_subject_raw"} and value not in {"", None}:
                value = f"{float(value):.4f}"
            formatted.append(str(value))
        formatted_rows.append(formatted)
    widths = [len(header) for header in headers]
    for row in formatted_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]
    lines = [" ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True))]
    lines.extend(" ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) for row in formatted_rows)
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
