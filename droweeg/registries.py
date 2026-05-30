from __future__ import annotations

from typing import Any, Callable


DATASET_REGISTRY: dict[str, Callable[..., Any]] = {}
MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}
METHOD_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_dataset(name: str, cls_or_factory: Callable[..., Any]) -> None:
    DATASET_REGISTRY[_normalize_name(name)] = cls_or_factory


def register_model(name: str, cls_or_factory: Callable[..., Any]) -> None:
    MODEL_REGISTRY[_normalize_name(name)] = cls_or_factory


def register_method(name: str, cls_or_factory: Callable[..., Any]) -> None:
    METHOD_REGISTRY[_normalize_name(name)] = cls_or_factory


def get_dataset(name: str) -> Callable[..., Any]:
    return _get(DATASET_REGISTRY, name, "dataset")


def get_model(name: str) -> Callable[..., Any]:
    return _get(MODEL_REGISTRY, name, "model")


def get_method(name: str) -> Callable[..., Any]:
    return _get(METHOD_REGISTRY, name, "method")


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def list_methods() -> list[str]:
    return sorted(METHOD_REGISTRY)


def _get(registry: dict[str, Callable[..., Any]], name: str, kind: str) -> Callable[..., Any]:
    normalized = _normalize_name(name)
    if normalized not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"Unknown {kind}: {name!r}. Available {kind}s: {available}")
    return registry[normalized]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()
