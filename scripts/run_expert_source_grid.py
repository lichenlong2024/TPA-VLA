"""Run an Expert-source cross-validation grid on cached hidden-state data.

Each row chooses the Phase-I Expert source. Each column chooses the target task
whose Query is trained against that source Expert. The output CSV reports cached
L1 validation loss; full LIBERO success-rate evaluation can use the saved Query
checkpoints with the same source Expert.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_eval_loss(path: Path) -> float:
    import json

    with path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["mean_l1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    python = sys.executable
    root = Path(cfg["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    tasks: List[Dict] = cfg["tasks"]
    hidden_dim = str(cfg.get("hidden_dim", 896))
    rows = []

    for source in tasks:
        source_name = source["name"]
        expert_dir = root / f"expert_source_{source_name}"
        phase1_steps = cfg.get("phase1_steps", 10000)
        run(
            [
                python,
                "scripts/train_phase1_expert.py",
                "--train_cache",
                source["phase1_cache"],
                "--output_dir",
                str(expert_dir),
                "--hidden_dim",
                hidden_dim,
                "--batch_size",
                str(cfg.get("batch_size", 16)),
                "--max_steps",
                str(phase1_steps),
                "--learning_rate",
                str(cfg.get("learning_rate", 1e-4)),
                "--num_blocks",
                str(cfg.get("expert_blocks", 24)),
            ],
            dry_run=args.dry_run,
        )
        expert_checkpoint = expert_dir / f"action_expert--step{phase1_steps}.pt"
        proprio_checkpoint = expert_dir / f"proprio_projector--step{phase1_steps}.pt"

        for target in tasks:
            target_name = target["name"]
            cell_dir = root / f"E_{source_name}__T_{target_name}"
            query_dir = cell_dir / "query"
            eval_json = cell_dir / "cached_eval.json"
            run(
                [
                    python,
                    "scripts/train_phase2_query.py",
                    "--train_cache",
                    target["phase2_cache"],
                    "--expert_checkpoint",
                    str(expert_checkpoint),
                    "--proprio_checkpoint",
                    str(proprio_checkpoint),
                    "--output_dir",
                    str(query_dir),
                    "--hidden_dim",
                    hidden_dim,
                    "--query_layers",
                    str(cfg.get("query_layers", 3)),
                    "--query_heads",
                    str(cfg.get("query_heads", 8)),
                    "--batch_size",
                    str(cfg.get("batch_size", 16)),
                    "--max_steps",
                    str(cfg.get("phase2_steps", 10000)),
                    "--learning_rate",
                    str(cfg.get("learning_rate", 1e-4)),
                    "--expert_blocks",
                    str(cfg.get("expert_blocks", 24)),
                ],
                dry_run=args.dry_run,
            )
            if target.get("eval_cache"):
                run(
                    [
                        python,
                        "scripts/eval_cached_policy.py",
                        "--eval_cache",
                        target["eval_cache"],
                        "--expert_checkpoint",
                        str(expert_checkpoint),
                        "--proprio_checkpoint",
                        str(proprio_checkpoint),
                        "--query_checkpoint",
                        str(query_dir / "query_module--final.pt"),
                        "--output_json",
                        str(eval_json),
                        "--hidden_dim",
                        hidden_dim,
                        "--expert_blocks",
                        str(cfg.get("expert_blocks", 24)),
                        "--query_layers",
                        str(cfg.get("query_layers", 3)),
                        "--query_heads",
                        str(cfg.get("query_heads", 8)),
                    ],
                    dry_run=args.dry_run,
                )
                metric = "" if args.dry_run else read_eval_loss(eval_json)
            else:
                metric = ""
            rows.append({"expert_source": source_name, "target_task": target_name, "cached_eval_l1": metric})

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["expert_source", "target_task", "cached_eval_l1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote grid summary: {out}")


if __name__ == "__main__":
    main()
