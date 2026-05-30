from __future__ import annotations

from collections.abc import Sequence


def run_backend(argv: Sequence[str] | None = None) -> None:
    import train_eegnet_source

    train_eegnet_source.main(None if argv is None else list(argv))
