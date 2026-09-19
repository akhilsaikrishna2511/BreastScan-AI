"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

import yaml


def deep_update(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating the input dictionaries."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, MutableMapping) and isinstance(merged.get(key), MutableMapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    return config


def resolve_path(path: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    """Resolve a possibly relative path against a base directory."""

    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved
