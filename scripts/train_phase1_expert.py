"""Phase I: train an ActionExpert from adaptable-backbone hidden states.

This script expects cached hidden states extracted from the temporarily adapted
VLM used in Phase I. It trains the Expert G and the proprio projector with an
action L1 objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tpa_vla.constants import ACTION_CHUNK_SIZE, ACTION_DIM, PROPRIO_DIM
from tpa_vla.data import HiddenStateActionDataset
from tpa_vla.modules import ActionExpert, ProprioProjector
from tpa_vla.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_cache", required=True, help="Path to Phase-I hidden-state cache (.pt/.npz).")
    parser.add_argument("--output_dir", required=True, help="Directory for expert checkpoints.")
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_blocks", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    dataset = HiddenStateActionDataset(args.train_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    iterator = iter(loader)

    expert = ActionExpert(
        input_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        action_dim=ACTION_DIM,
        action_chunk_size=ACTION_CHUNK_SIZE,
        num_blocks=args.num_blocks,
    ).to(device=device, dtype=dtype)
    proprio_projector = ProprioProjector(llm_dim=args.hidden_dim, proprio_dim=PROPRIO_DIM).to(device=device, dtype=dtype)

    optimizer = AdamW(
        list(expert.parameters()) + list(proprio_projector.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    progress = tqdm(range(1, args.max_steps + 1), dynamic_ncols=True, desc="phase1-expert")
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        hidden_states = batch["hidden_states"].to(device=device, dtype=dtype)
        proprio = batch["proprio"].to(device=device, dtype=dtype)
        actions = batch["actions"].to(device=device, dtype=dtype)

        pred = expert.predict_action(hidden_states, proprio, proprio_projector)
        loss = F.l1_loss(pred, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(expert.parameters()) + list(proprio_projector.parameters()), 1.0)
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.save_every == 0 or step == args.max_steps:
            torch.save(expert.state_dict(), out_dir / f"action_expert--step{step}.pt")
            torch.save(proprio_projector.state_dict(), out_dir / f"proprio_projector--step{step}.pt")

    with (out_dir / "phase1_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
