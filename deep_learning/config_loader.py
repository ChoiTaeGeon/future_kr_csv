"""
Configuration Loader & Validator for Deep Learning Pipeline
Loads configs/config.yaml and merges with CLI parameters.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

def load_config(config_path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load yaml configuration file, apply optional dictionary overrides,
    and validate required fields.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v

    # Required structure validation
    required_keys = ["data_dir", "tick_window", "coarse_to_fine", "cost_model", "model_save_dir", "progress_display"]
    for rk in required_keys:
        if rk not in cfg:
            raise ValueError(f"Missing required configuration key: '{rk}' in {path}")

    tw = cfg["tick_window"]
    for param in ["start", "end", "step"]:
        if param not in tw:
            raise ValueError(f"tick_window must contain '{param}' parameter")

    return cfg
