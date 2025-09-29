"""
Minimal Hydra-free instantiation helpers.

These functions allow constructing Python objects from OmegaConf configs that
use the common "_target_"/"_partial_" schema.
"""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any

from omegaconf import DictConfig, ListConfig


_SPECIAL_KEYS = {"_target_", "_partial_"}


def _locate_target(target_path: str) -> Any:
    """Import and return the callable at the given dotted path."""
    if ":" in target_path:
        module_path, attr_path = target_path.split(":", 1)
    else:
        parts = target_path.split(".")
        module_path, attr_path = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_path)
    target = module
    for attr in attr_path.split("."):
        target = getattr(target, attr)
    return target


def instantiate(config: Any, **kwargs: Any) -> Any:
    """Instantiate objects from OmegaConf configs without Hydra.

    - Dicts with "_target_" key are turned into callables/instances.
    - Lists and dicts are recursively processed.
    - If "_partial_" is True, returns a partially-applied callable.
    """
    # OmegaConf containers
    if isinstance(config, DictConfig):
        cfg_dict = {k: v for k, v in config.items()}
        if "_target_" in cfg_dict:
            target = _locate_target(cfg_dict["_target_"])
            is_partial = bool(cfg_dict.get("_partial_", False))
            # Recursively instantiate arguments
            args = {
                k: instantiate(v) for k, v in cfg_dict.items() if k not in _SPECIAL_KEYS
            }
            args.update(kwargs)
            return partial(target, **args) if is_partial else target(**args)
        # Plain dictconfig: recurse
        return {k: instantiate(v) for k, v in cfg_dict.items()}

    if isinstance(config, ListConfig):
        return [instantiate(v) for v in list(config)]

    # Native containers
    if isinstance(config, dict):
        if "_target_" in config:
            target = _locate_target(config["_target_"])
            is_partial = bool(config.get("_partial_", False))
            args = {k: instantiate(v) for k, v in config.items() if k not in _SPECIAL_KEYS}
            args.update(kwargs)
            return partial(target, **args) if is_partial else target(**args)
        return {k: instantiate(v) for k, v in config.items()}

    if isinstance(config, (list, tuple)):
        return type(config)(instantiate(v) for v in config)

    # Base case
    return config


