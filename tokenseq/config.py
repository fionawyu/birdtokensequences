"""Configuration loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_TASKS = REPO_ROOT / "configs" / "tasks.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def load_tasks(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    with open(path or DEFAULT_TASKS) as f:
        return yaml.safe_load(f)["tasks"]


def with_overrides(cfg: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with section-level overrides, ignoring None values."""
    out = copy.deepcopy(cfg)
    for section, values in sections.items():
        for key, value in values.items():
            if value is not None:
                out[section][key] = value
    return out
