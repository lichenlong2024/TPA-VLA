"""Run the public TPA-VLA component pipeline from a YAML config.

This orchestrates the reviewer-facing reproduction path:
  1. train Phase-I Expert on adapted-backbone hidden states,
  2. train one Phase-II Query per task on frozen-backbone hidden states,
  3. optionally evaluate each Query on cached hidden-state validation data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    python = sys.executable
    root = Path(cfg["run_root"])
    root.mkdir(parents=True, exist_ok=True)

    phase1 = cfg["phase1"]
    expert_dir = root / "shared_expert"
    run_command(
        [
            python,
            "scripts/train_phase1_expert.py",
            "--train_cache",
            phase1["train_cache"],
            "--output_dir",
            str(expert_dir),
            "--hidden_dim",
            str(cfg.get("hidden_dim", 896)),
            "--batch_size",
            str(cfg.get("batch_size", 16)),
            "--max_steps",
            str(cfg.get("phase1_steps", 10000)),
            "--learning_rate",
            str(cfg.get("learning_rate", 1e-4)),
            "--num_blocks",
            str(cfg.get("expert_blocks", 24)),
        ],
        dry_run=args.dry_run,
    )

    expert_checkpoint = expert_dir / f"action_expert--step{cfg.get('phase1_steps', 10000)}.pt"
    proprio_checkpoint = expert_dir / f"proprio_projector--step{cfg.get('phase1_steps', 10000)}.pt"

    for task in cfg["tasks"]:
        task_name = task["name"]
        query_dir = root / task_name / "query"
        run_command(
            [
                python,
                "scripts/train_phase2_query.py",
                "--train_cache",
                task["train_cache"],
                "--expert_checkpoint",
                str(expert_checkpoint),
                "--proprio_checkpoint",
                str(proprio_checkpoint),
                "--output_dir",
                str(query_dir),
                "--hidden_dim",
                str(cfg.get("hidden_dim", 896)),
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

        if task.get("eval_cache"):
            run_command(
                [
                    python,
                    "scripts/eval_cached_policy.py",
                    "--eval_cache",
                    task["eval_cache"],
                    "--expert_checkpoint",
                    str(expert_checkpoint),
                    "--proprio_checkpoint",
                    str(proprio_checkpoint),
                    "--query_checkpoint",
                    str(query_dir / "query_module--final.pt"),
                    "--output_json",
                    str(root / task_name / "cached_eval.json"),
                    "--hidden_dim",
                    str(cfg.get("hidden_dim", 896)),
                    "--expert_blocks",
                    str(cfg.get("expert_blocks", 24)),
                    "--query_layers",
                    str(cfg.get("query_layers", 3)),
                    "--query_heads",
                    str(cfg.get("query_heads", 8)),
                ],
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
