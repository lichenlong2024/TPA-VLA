"""Dataset helpers for TPA-VLA component training.

The scripts in this reviewer-facing repository train the method-specific
components from cached VLM hidden states. This keeps the release focused on the
TPA-VLA mechanism while allowing users to connect any compatible VLA backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class HiddenStateActionDataset(Dataset):
    """Load cached hidden states, proprio states, and action chunks.

    Supported file formats:
      - `.pt` or `.pth`: a dictionary with `hidden_states`, `proprio`, `actions`
      - `.npz`: arrays with the same three keys

    Shapes:
      - hidden_states: [N, layers, tokens, hidden_dim]
      - proprio: [N, proprio_dim]
      - actions: [N, chunk, action_dim]
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        payload = self._load_payload(self.path)
        self.hidden_states = self._as_tensor(payload["hidden_states"], torch.float32)
        self.proprio = self._as_tensor(payload["proprio"], torch.float32)
        self.actions = self._as_tensor(payload["actions"], torch.float32)
        self._validate()

    def _load_payload(self, path: Path) -> Dict[str, Any]:
        if path.suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu")
        elif path.suffix == ".npz":
            with np.load(path) as npz:
                payload = {key: npz[key] for key in npz.files}
        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .pt, .pth, or .npz.")
        missing = {"hidden_states", "proprio", "actions"} - set(payload)
        if missing:
            raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
        return payload

    @staticmethod
    def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    def _validate(self) -> None:
        if self.hidden_states.ndim != 4:
            raise ValueError(f"hidden_states must be [N, L, T, D], got {tuple(self.hidden_states.shape)}")
        if self.proprio.ndim != 2:
            raise ValueError(f"proprio must be [N, proprio_dim], got {tuple(self.proprio.shape)}")
        if self.actions.ndim != 3:
            raise ValueError(f"actions must be [N, chunk, action_dim], got {tuple(self.actions.shape)}")
        n = self.hidden_states.shape[0]
        if self.proprio.shape[0] != n or self.actions.shape[0] != n:
            raise ValueError("hidden_states, proprio, and actions must have the same first dimension")

    def __len__(self) -> int:
        return int(self.hidden_states.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "hidden_states": self.hidden_states[idx],
            "proprio": self.proprio[idx],
            "actions": self.actions[idx],
        }
