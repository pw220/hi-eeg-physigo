from __future__ import annotations

from typing import Any

from droweeg.datasets.base import EEGDataset
from droweeg.datasets.sadt_balanced import SADTBalancedDataset
from droweeg.datasets.seedvig import SeedVIGDataset
from droweeg.datasets.standard_npz import StandardDataset, save_standard_dataset
from droweeg.methods.source_only import SourceOnlyMethod
from droweeg.models.eegnet import EEGNet
from droweeg.registries import (
    get_dataset,
    get_method,
    get_model,
    list_datasets,
    list_methods,
    list_models,
    register_dataset,
    register_method,
    register_model,
)

register_dataset("seedvig", SeedVIGDataset)
register_dataset("sadt-balanced", SADTBalancedDataset)
register_dataset("standard-npz", StandardDataset)
register_model("eegnet", EEGNet)
register_method("source_only", SourceOnlyMethod)


def model(name: str, **kwargs):
    dataset_obj = kwargs.pop("dataset", None)
    if dataset_obj is not None:
        metadata = dataset_obj.get_metadata()
        kwargs.setdefault("channels", metadata["input_channels"])
        kwargs.setdefault("samples", metadata["input_samples"])
        kwargs.setdefault("num_classes", metadata["num_classes"])
    return get_model(name)(**kwargs)


def dataset(name: str, **kwargs):
    return get_dataset(name)(**kwargs)


def method(name: str, **kwargs):
    return get_method(name)(**kwargs)


def run(**kwargs: Any):
    from droweeg.train import run_from_kwargs

    return run_from_kwargs(**kwargs)


__all__ = [
    "Dataset",
    "EEGDataset",
    "dataset",
    "get_dataset",
    "get_method",
    "get_model",
    "list_datasets",
    "list_methods",
    "list_models",
    "method",
    "model",
    "register_dataset",
    "register_method",
    "register_model",
    "run",
    "save_standard_dataset",
]


Dataset = EEGDataset
