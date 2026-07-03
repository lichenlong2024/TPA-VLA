"""Phase II: train a task QueryModule while keeping the Expert fixed.

This script takes cached frozen-backbone hidden states and a Phase-I Expert. It
optimizes only the QueryModule, using action-level supervision through the fixed
Expert rather than feature distillation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpa_vla.data import HiddenStateActionDataset
from tpa_vla.modules import ActionExpert, ProprioProjector, QueryModule, QueryWrappedExpert
from tpa_vla.utils import ensure_dir, load_component_state_dict, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_cache", required=True, help="Path to frozen-backbone hidden-state cache (.pt/.npz).")
    parser.add_argument("--expert_checkpoint", required=True, help="Path to Phase-I ActionExpert checkpoint.")
    parser.add_argument("--proprio_checkpoint", required=True, help="Path to Phase-I ProprioProjector checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory for QueryModule checkpoints.")
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--query_layers", type=int, default=3)
    parser.add_argument("--query_heads", type=int, default=8)
    parser.add_argument("--query_dropout", type=float, default=0.1)
    parser.add_argument("--expert_blocks", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
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

    expert = ActionExpert(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_blocks=args.expert_blocks)
    expert.load_state_dict(load_component_state_dict(args.expert_checkpoint), strict=True)
    expert.to(device=device, dtype=dtype).eval()
    for param in expert.parameters():
        param.requires_grad_(False)

    proprio_projector = ProprioProjector(llm_dim=args.hidden_dim)
    proprio_projector.load_state_dict(load_component_state_dict(args.proprio_checkpoint), strict=True)
    proprio_projector.to(device=device, dtype=dtype).eval()
    for param in proprio_projector.parameters():
        param.requires_grad_(False)

    query = QueryModule(
        input_dim=args.hidden_dim,
        num_heads=args.query_heads,
        num_transformer_layers=args.query_layers,
        dropout=args.query_dropout,
        output_dim=args.hidden_dim,
    ).to(device=device, dtype=dtype)
    policy = QueryWrappedExpert(query, expert)
    optimizer = AdamW(query.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    progress = tqdm(range(1, args.max_steps + 1), dynamic_ncols=True, desc="phase2-query")
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        hidden_states = batch["hidden_states"].to(device=device, dtype=dtype)
        proprio = batch["proprio"].to(device=device, dtype=dtype)
        actions = batch["actions"].to(device=device, dtype=dtype)

        pred = policy.predict_action(hidden_states, proprio, proprio_projector)
        loss = F.l1_loss(pred, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query.parameters(), 1.0)
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.save_every == 0 or step == args.max_steps:
            torch.save(query.state_dict(), out_dir / f"query_module--step{step}.pt")

    torch.save(query.state_dict(), out_dir / "query_module--final.pt")
    with (out_dir / "query_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
