from __future__ import annotations


class Method:
    name = "base"

    def fit_source(self, *args, **kwargs):
        raise NotImplementedError

    def adapt_target(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def save_outputs(self, *args, **kwargs):
        raise NotImplementedError
