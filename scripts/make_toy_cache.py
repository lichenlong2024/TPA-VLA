"""Create a tiny synthetic hidden-state cache for smoke testing.

The generated cache is not a benchmark dataset. It only verifies that the
released TPA-VLA modules and scripts run end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpa_vla.constants import ACTION_CHUNK_SIZE, ACTION_DIM, DEFAULT_TASK_TOKENS, PROPRIO_DIM
from tpa_vla.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    seq_len = DEFAULT_TASK_TOKENS + ACTION_DIM
    hidden_states = torch.randn(args.num_samples, args.num_layers, seq_len, args.hidden_dim)
    proprio = torch.randn(args.num_samples, PROPRIO_DIM)

    # Make the target deterministic enough for a smoke test without encoding a
    # meaningful robot task.
    base = hidden_states[:, -1, -ACTION_DIM:, :].mean(dim=(1, 2))
    actions = torch.tanh(base).view(-1, 1, 1).repeat(1, ACTION_CHUNK_SIZE, ACTION_DIM)
    actions = actions + 0.01 * torch.randn_like(actions)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden_states": hidden_states, "proprio": proprio, "actions": actions}, out)
    print(f"Wrote toy cache: {out}")


if __name__ == "__main__":
    main()
