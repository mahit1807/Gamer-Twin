"""
Utility functions for GAMER-TWIN: configuration, seeding, logging, and metrics.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Global Project Target Classes (Nine Selected Classes)
PROJECT_CLASSES = [
    "Fire",
    "Reload",
    "Switch Weapon",
    "Inspect",
    "Scope",
    "Grenade",
    "Jump",
    "Crouch",
    "Move"
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(PROJECT_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(PROJECT_CLASSES)}


def setup_logger(name: str = "GAMER-TWIN", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads project configuration from YAML file.
    Falls back to a default configuration dict if YAML loading fails.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = str(base_dir / "config" / "config.yaml")

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Could not load YAML config from {config_path} ({e}). Using default config.")

    # Robust default configuration
    return {
        "project": {
            "name": "GAMER-TWIN",
            "random_seed": 42
        },
        "data": {
            "raw_dir": "data/raw",
            "sample_dir": "data/sample_data",
            "processed_dir": "outputs/processed",
            "target_classes": PROJECT_CLASSES
        },
        "sequence": {
            "window_size": 16,
            "stride": 2,
            "target_offset": 1
        },
        "split": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "split_strategy": "round_id"
        },
        "models": {
            "common": {"hidden_dim": 64, "dropout": 0.1, "num_classes": 9},
            "lstm_baseline": {"num_layers": 2},
            "lstm_transformer": {"lstm_layers": 1, "transformer_layers": 1, "num_heads": 4, "dim_feedforward": 128},
            "transformer": {"num_layers": 2, "num_heads": 4, "dim_feedforward": 128},
            "transformer_ssl": {"num_layers": 2, "num_heads": 4, "dim_feedforward": 128, "ssl_mask_prob": 0.20}
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "early_stopping_patience": 4,
            "device": "auto"
        },
        "evaluation": {
            "top_k": 3
        }
    }


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """
    Returns torch device string ('cuda' or 'cpu').
    Gracefully falls back to CPU if CUDA is unavailable.
    """
    if preference == "cpu":
        return "cpu"
    try:
        import torch
        if preference in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def save_json(data: Any, file_path: str) -> None:
    """Saves arbitrary data structure to formatted JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Any:
    """Loads JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
