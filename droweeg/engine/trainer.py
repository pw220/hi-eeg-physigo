from __future__ import annotations

from collections.abc import Sequence


def run_backend(argv: Sequence[str] | None = None) -> None:
    from droweeg.engine import sourceonly_backend

    sourceonly_backend.main(None if argv is None else list(argv))


def fit_source(*args, **kwargs):
    """Compatibility boundary for source fitting.

    The current implementation delegates to the stable legacy backend functions
    to preserve source-only numbers during Phase 0.
    """

    from droweeg.engine.sourceonly_backend import train_one_epoch

    return train_one_epoch(*args, **kwargs)


def train_one_epoch(*args, **kwargs):
    from droweeg.engine.sourceonly_backend import train_one_epoch as _train_one_epoch

    return _train_one_epoch(*args, **kwargs)
