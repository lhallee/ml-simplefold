"""
Lightweight config loader to replace Hydra's defaults mechanism for this repo.

Supports:
- Loading a base YAML file (e.g., configs/base_train.yaml)
- Processing its `defaults` list to include group files
- Simple variable interpolation with ${a.b.c} via OmegaConf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import OmegaConf


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_defaults(root: Path, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = list(base_cfg.pop("defaults", []) or [])
    composed: Dict[str, Any] = {}
    for entry in defaults:
        if entry is None:
            continue
        if isinstance(entry, str):
            # group name (e.g., "data: pdb" is handled as dict below)
            continue
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key == "_self_":
                    # ignored: our base file is already loaded
                    continue
                # entry like {"data": "pdb"} or {"model/architecture": "foldingdit_100M"}
                group = key
                name = value
                group_path = root / group
                if not group_path.suffix:
                    # group directory
                    cfg_path = group_path / f"{name}.yaml"
                else:
                    # explicit file
                    cfg_path = group_path
                if not cfg_path.exists():
                    raise FileNotFoundError(f"Config not found: {cfg_path}")
                group_cfg = _load_yaml(cfg_path)
                composed = _merge_dicts(composed, group_cfg)

    # finally merge base on top (so base can override included groups similar to Hydra's order with _self_ last)
    composed = _merge_dicts(composed, base_cfg)
    return composed


def load_config(config_path: str) -> Any:
    """Load and compose a config starting from a base YAML file."""
    base_path = Path(config_path)
    root = base_path.parent
    base_cfg = _load_yaml(base_path)
    composed = _resolve_defaults(root, base_cfg)
    # Use OmegaConf for interpolation and structured containers
    cfg = OmegaConf.create(composed)
    OmegaConf.resolve(cfg)
    return cfg


