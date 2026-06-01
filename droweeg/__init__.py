from __future__ import annotations

from typing import Any

from droweeg.datasets.base import EEGDataset
from droweeg.datasets.standard_npz import StandardDataset, make_toy_dataset, save_standard_dataset
from droweeg.registries import (
    get_dataset,
    get_method,
    get_model,
    list_datasets as _registry_list_datasets,
    list_methods,
    list_models,
    register_builtin_components,
    register_dataset,
    register_method,
    register_model,
)
from droweeg.results import DrowEEGResults

register_builtin_components()


def model(name: str, **kwargs):
    data_path = kwargs.pop("data", None)
    if data_path is not None and "dataset" not in kwargs:
        kwargs["dataset"] = load_dataset(data_path)
    dataset_obj = kwargs.pop("dataset", None)
    if dataset_obj is not None:
        metadata = dataset_obj.get_metadata()
        kwargs.setdefault("channels", metadata["input_channels"])
        kwargs.setdefault("samples", metadata["input_samples"])
        kwargs.setdefault("num_classes", metadata["num_classes"])
    return get_model(name)(**kwargs)


def dataset(name: str, **kwargs):
    return get_dataset(name)(**kwargs)


def list_datasets(*, include_internal: bool = False) -> list[str]:
    names = _registry_list_datasets()
    if include_internal:
        return names
    return [name for name in names if name != "standard-npz"]


def load_dataset(path: str, **kwargs):
    return StandardDataset(path=path, **kwargs).load()


def method(name: str, **kwargs):
    return get_method(name)(**kwargs)


def run(**kwargs: Any):
    from droweeg.train import run_from_kwargs

    return run_from_kwargs(**kwargs)


__all__ = [
    "Dataset",
    "DrowEEGResults",
    "EEGDataset",
    "dataset",
    "get_dataset",
    "get_method",
    "get_model",
    "list_datasets",
    "list_methods",
    "list_models",
    "load_dataset",
    "make_toy_dataset",
    "method",
    "model",
    "register_dataset",
    "register_method",
    "register_model",
    "run",
    "save_standard_dataset",
]


Dataset = StandardDataset
