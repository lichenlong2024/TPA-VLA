"""Evaluate a TPA-VLA policy on a cached hidden-state action dataset.

For full LIBERO rollout evaluation, connect the same QueryWrappedExpert module
to the OpenVLA/OpenVLA-OFT LIBERO evaluation loop. This lightweight evaluator is
provided for quick checkpoint validation without simulator dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpa_vla.data import HiddenStateActionDataset
from tpa_vla.modules import ActionExpert, ProprioProjector, QueryModule, QueryWrappedExpert
from tpa_vla.utils import load_component_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_cache", required=True)
    parser.add_argument("--expert_checkpoint", required=True)
    parser.add_argument("--proprio_checkpoint", required=True)
    parser.add_argument("--query_checkpoint", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--expert_blocks", type=int, default=24)
    parser.add_argument("--query_layers", type=int, default=3)
    parser.add_argument("--query_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    expert = ActionExpert(input_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_blocks=args.expert_blocks)
    expert.load_state_dict(load_component_state_dict(args.expert_checkpoint), strict=True)
    expert.to(device=device, dtype=dtype).eval()

    proprio_projector = ProprioProjector(llm_dim=args.hidden_dim)
    proprio_projector.load_state_dict(load_component_state_dict(args.proprio_checkpoint), strict=True)
    proprio_projector.to(device=device, dtype=dtype).eval()

    query = QueryModule(input_dim=args.hidden_dim, num_heads=args.query_heads, num_transformer_layers=args.query_layers)
    query.load_state_dict(load_component_state_dict(args.query_checkpoint), strict=True)
    query.to(device=device, dtype=dtype).eval()
    policy = QueryWrappedExpert(query, expert)

    dataset = HiddenStateActionDataset(args.eval_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    losses = []
    for batch in tqdm(loader, dynamic_ncols=True, desc="eval-cached-policy"):
        pred = policy.predict_action(
            batch["hidden_states"].to(device=device, dtype=dtype),
            batch["proprio"].to(device=device, dtype=dtype),
            proprio_projector,
        )
        target = batch["actions"].to(device=device, dtype=dtype)
        losses.append(float(F.l1_loss(pred, target).cpu()))

    metrics = {"num_batches": len(losses), "mean_l1": sum(losses) / max(1, len(losses))}
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
