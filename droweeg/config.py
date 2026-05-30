from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - clear runtime dependency message
        raise RuntimeError("YAML config support requires PyYAML. Install requirements.txt.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def kwargs_to_argv(kwargs: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv
