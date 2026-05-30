from __future__ import annotations

from .base import Method


class SourceOnlyMethod(Method):
    """Source-only baseline wrapper.

    This method delegates to the current source-only trainer. It does not use
    target labels for training, validation, normalization, clipping, class
    weighting, checkpoint selection, or model selection.
    """

    name = "source_only"

    def fit_source(self, *args, **kwargs):
        raise RuntimeError("Use droweeg.run(...) or python -m droweeg.train for this compatibility wrapper.")

    def adapt_target(self, *args, **kwargs):
        return None

    def evaluate(self, *args, **kwargs):
        raise RuntimeError("Use droweeg.run(...) or python -m droweeg.train for this compatibility wrapper.")

    def save_outputs(self, *args, **kwargs):
        raise RuntimeError("Use droweeg.run(...) or python -m droweeg.train for this compatibility wrapper.")
