from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra.utils import to_absolute_path

from src import PROJECT_ROOT


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_path(path_like) -> Path:
    return Path(to_absolute_path(str(path_like)))


def prepare_checkpoint_path(config) -> Path:
    path = resolve_path(Path(config.trainer.save_dir) / config.trainer.save_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: Optional[str] = None) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
