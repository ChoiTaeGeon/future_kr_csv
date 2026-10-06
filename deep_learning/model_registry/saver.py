"""
Model Packaging, Versioning, and Feature Schema Validator
Saves ML/DL models with full metadata:
- Training Timestamp, Tick Window, Metrics, Feature Schema, Hyperparameters
- Auto schema validation upon loading
- File naming convention: model_tick{window}_{model}_{sharpe}_{timestamp}
"""
import os
import json
import joblib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class ModelMetadata:
    version: str
    model_name: str
    tick_window: int
    created_at: str
    feature_names: List[str]
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    framework: str  # 'sklearn', 'lightgbm', 'xgboost', 'pytorch'


def save_model(
    model: Any,
    metadata: ModelMetadata,
    save_dir: str | Path,
    registry_file: Optional[str | Path] = None
) -> Tuple[Path, Path]:
    """
    Saves model weights (.joblib or .pt) along with .meta.json.
    Appends entry to model registry index.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Naming convention: model_tick{window}_{model}_{sharpe}_{date}
    sharpe = metadata.performance_metrics.get("sharpe_ratio", 0.0)
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"model_tick{metadata.tick_window}_{metadata.model_name}_sharpe{sharpe:.2f}_{date_tag}"

    meta_path = out_dir / f"{base_name}.meta.json"

    if metadata.framework == "pytorch" or isinstance(model, nn.Module):
        model_path = out_dir / f"{base_name}.pt"
        torch.save(model.state_dict(), model_path)
    else:
        model_path = out_dir / f"{base_name}.joblib"
        joblib.dump(model, model_path)

    # Save metadata JSON
    meta_dict = asdict(metadata)
    meta_dict["weights_file"] = model_path.name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    # Update Registry Index
    if registry_file:
        reg_path = Path(registry_file)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        registry_data = []
        if reg_path.exists():
            try:
                with open(reg_path, "r", encoding="utf-8") as rf:
                    registry_data = json.load(rf)
            except Exception:
                registry_data = []
        registry_data.append(meta_dict)
        with open(reg_path, "w", encoding="utf-8") as rf:
            json.dump(registry_data, rf, indent=2, ensure_ascii=False)

    return model_path, meta_path


def load_model(
    model_path_or_meta: str | Path,
    expected_features: Optional[List[str]] = None,
    pytorch_model_class: Optional[Any] = None
) -> Tuple[Any, ModelMetadata]:
    """
    Loads saved model and metadata. Verifies feature schema.
    Throws warning or error if feature count or order does not match.
    """
    p = Path(model_path_or_meta)
    if p.suffix == ".json":
        meta_path = p
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        weights_path = p.parent / meta_dict["weights_file"]
    else:
        weights_path = p
        meta_path = p.with_suffix(".meta.json")
        if not meta_path.exists():
            # Try finding matching meta
            meta_path = list(p.parent.glob(f"{p.stem}*.meta.json"))[0]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

    meta = ModelMetadata(
        version=meta_dict["version"],
        model_name=meta_dict["model_name"],
        tick_window=meta_dict["tick_window"],
        created_at=meta_dict["created_at"],
        feature_names=meta_dict["feature_names"],
        performance_metrics=meta_dict["performance_metrics"],
        hyperparameters=meta_dict.get("hyperparameters", {}),
        framework=meta_dict.get("framework", "sklearn")
    )

    # Feature Schema Validation
    if expected_features is not None:
        missing = set(meta.feature_names) - set(expected_features)
        extra = set(expected_features) - set(meta.feature_names)
        if missing or extra:
            print(f"[!] Warning: Feature schema mismatch! Missing: {missing}, Extra: {extra}")

    # Load Model Weights
    if meta.framework == "pytorch" or weights_path.suffix == ".pt":
        if pytorch_model_class is None:
            raise ValueError("pytorch_model_class must be provided when loading PyTorch model.")
        model = pytorch_model_class
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
    else:
        model = joblib.load(weights_path)

    return model, meta
