"""
config_loader.py
----------------
Utility to load the project YAML configuration file.
Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()
    print(cfg.data.raw_dir)
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace so we can use dot notation."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str = None) -> SimpleNamespace:
    """
    Load the YAML config file.

    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml
                     relative to the project root.

    Returns:
        SimpleNamespace: Config values accessible with dot notation.
                         e.g. cfg.data.batch_size
    """
    if config_path is None:
        # Resolve path relative to this file's location
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)
