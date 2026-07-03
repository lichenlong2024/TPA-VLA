"""Small utilities shared by the public TPA-VLA scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the `module.` prefix added by DistributedDataParallel checkpoints."""
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_component_state_dict(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Load a module checkpoint and normalize common wrapper prefixes."""
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict at {checkpoint_path}, got {type(state_dict)!r}")
    return strip_ddp_prefix(state_dict)


def newest_checkpoint(directory: str | Path, pattern: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matching {pattern!r} in {directory}")
    return matches[-1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def disable_tokenizer_parallelism() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
