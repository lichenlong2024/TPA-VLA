"""Task-switching overhead microbenchmark for TPA-VLA.

This script isolates the cost of making a newly requested task available under
a low-memory dynamic-serving setting. It excludes simulator stepping, network
communication, image preprocessing, and VLM forward cost. Each request uses
fixed dummy VLM hidden states and measures:

  request arrives -> task-specific modules are loaded -> first action output

Compared modes:
  baseline
      Switch a task-specific VLA-side adapter proxy, Expert, and ProprioProjector.
  task_spec_expert
      Switch a task-specific Query, Expert, and ProprioProjector.
  tpa_vla
      Keep one Expert and ProprioProjector resident; switch only the Query.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpa_vla.constants import ACTION_DIM, DEFAULT_TASK_TOKENS, PROPRIO_DIM
from tpa_vla.modules import ActionExpert, ProprioProjector, QueryModule, QueryWrappedExpert
from tpa_vla.utils import load_component_state_dict, newest_checkpoint


@dataclass
class TaskPaths:
    task_id: int
    expert_checkpoint: Path
    proprio_checkpoint: Path
    query_checkpoint: Path
    adapter_checkpoint: Optional[Path] = None


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now_ms() -> float:
    cuda_sync()
    return time.perf_counter() * 1000.0


def discover_task_paths(run_root: Path, task_ids: Iterable[int]) -> List[TaskPaths]:
    tasks = []
    for task_id in task_ids:
        task_dir = run_root / f"task_{task_id}"
        expert_dir = task_dir / "expert"
        query_dir = task_dir / "query"
        adapter_dir = task_dir / "adapter"
        adapter_ckpt = newest_checkpoint(adapter_dir, "*.pt") if adapter_dir.exists() else None
        tasks.append(
            TaskPaths(
                task_id=task_id,
                expert_checkpoint=newest_checkpoint(expert_dir, "action_expert--*.pt"),
                proprio_checkpoint=newest_checkpoint(expert_dir, "proprio_projector--*.pt"),
                query_checkpoint=newest_checkpoint(query_dir, "query_module--*.pt"),
                adapter_checkpoint=adapter_ckpt,
            )
        )
    return tasks


def load_expert(path: Path, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> ActionExpert:
    expert = ActionExpert(input_dim=hidden_dim, hidden_dim=hidden_dim)
    expert.load_state_dict(load_component_state_dict(path), strict=True)
    return expert.to(device=device, dtype=dtype).eval()


def load_proprio(path: Path, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> ProprioProjector:
    projector = ProprioProjector(llm_dim=hidden_dim)
    projector.load_state_dict(load_component_state_dict(path), strict=True)
    return projector.to(device=device, dtype=dtype).eval()


def load_query(path: Path, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> QueryModule:
    query = QueryModule(input_dim=hidden_dim)
    query.load_state_dict(load_component_state_dict(path), strict=True)
    return query.to(device=device, dtype=dtype).eval()


def load_adapter_proxy(path: Optional[Path], device: torch.device) -> Dict[str, torch.Tensor]:
    """Load adapter tensors to GPU as a proxy for task-specific VLA-side state."""
    if path is None:
        return {}
    tensors = load_component_state_dict(path)
    return {key: value.to(device=device, non_blocking=True) for key, value in tensors.items() if torch.is_tensor(value)}


def release(resident: Dict[str, object]) -> None:
    resident.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def time_first_action(policy: nn.Module, proprio_projector: nn.Module, hidden: torch.Tensor, proprio: torch.Tensor) -> float:
    start = now_ms()
    if isinstance(policy, QueryWrappedExpert):
        _ = policy.predict_action(hidden, proprio, proprio_projector)
    else:
        _ = policy.predict_action(hidden, proprio, proprio_projector)
    return now_ms() - start


def build_sequence(task_ids: List[int], warmup_rounds: int, measure_rounds: int) -> List[Dict[str, object]]:
    sequence = []
    for phase, rounds in (("warmup", warmup_rounds), ("measure", measure_rounds)):
        for _ in range(rounds):
            for task_id in task_ids:
                sequence.append({"phase": phase, "task_id": task_id})
    return sequence


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a meaningful switching benchmark.")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    task_ids = args.task_ids or list(range(1, args.num_tasks + 1))
    tasks = discover_task_paths(Path(args.run_root), task_ids)
    task_by_id = {task.task_id: task for task in tasks}
    sequence = build_sequence(task_ids, args.warmup_rounds, args.measure_rounds)

    hidden = torch.randn(
        1,
        args.num_vlm_layers,
        DEFAULT_TASK_TOKENS + ACTION_DIM,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    )
    proprio = torch.zeros(1, PROPRIO_DIM, device=device, dtype=dtype)

    shared_task = task_by_id[args.shared_expert_task or task_ids[0]]
    shared_expert = load_expert(shared_task.expert_checkpoint, args.hidden_dim, device, dtype)
    shared_proprio = load_proprio(shared_task.proprio_checkpoint, args.hidden_dim, device, dtype)
    rows: List[Dict[str, object]] = []

    for mode in args.modes:
        resident: Dict[str, object] = {}
        current_task = None
        for step, item in enumerate(sequence):
            task_id = int(item["task_id"])
            task = task_by_id[task_id]
            switch_required = current_task != task_id
            cold_start = current_task is None

            switch_start = now_ms()
            if switch_required:
                release(resident)
                if mode == "baseline":
                    resident["adapter"] = load_adapter_proxy(task.adapter_checkpoint, device)
                    resident["expert"] = load_expert(task.expert_checkpoint, args.hidden_dim, device, dtype)
                    resident["proprio"] = load_proprio(task.proprio_checkpoint, args.hidden_dim, device, dtype)
                    resident["policy"] = resident["expert"]
                elif mode == "task_spec_expert":
                    resident["query"] = load_query(task.query_checkpoint, args.hidden_dim, device, dtype)
                    resident["expert"] = load_expert(task.expert_checkpoint, args.hidden_dim, device, dtype)
                    resident["proprio"] = load_proprio(task.proprio_checkpoint, args.hidden_dim, device, dtype)
                    resident["policy"] = QueryWrappedExpert(resident["query"], resident["expert"]).eval()
                elif mode == "tpa_vla":
                    resident["query"] = load_query(task.query_checkpoint, args.hidden_dim, device, dtype)
                    resident["expert"] = shared_expert
                    resident["proprio"] = shared_proprio
                    resident["policy"] = QueryWrappedExpert(resident["query"], resident["expert"]).eval()
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                current_task = task_id
            switch_ms = now_ms() - switch_start
            first_action_ms = time_first_action(resident["policy"], resident["proprio"], hidden, proprio)
            rows.append(
                {
                    "mode": mode,
                    "phase": item["phase"],
                    "step": step,
                    "task_id": task_id,
                    "switch_required": int(switch_required),
                    "cold_start": int(cold_start),
                    "switch_ms": switch_ms,
                    "first_action_ms": first_action_ms,
                    "switch_plus_first_action_ms": switch_ms + first_action_ms,
                    "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                }
            )
        release(resident)
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for mode in sorted({row["mode"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["mode"] == mode and row["phase"] == "measure" and row["switch_required"] and not row["cold_start"]
        ]
        for metric in ["switch_ms", "first_action_ms", "switch_plus_first_action_ms"]:
            values = sorted(float(row[metric]) for row in selected)
            if not values:
                continue
            p95 = values[max(0, min(len(values) - 1, math.ceil(0.95 * len(values)) - 1))]
            summary.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p95": p95,
                    "min": values[0],
                    "max": values[-1],
                }
            )
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, help="Root containing task_1/.../task_N checkpoints.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--task_ids", nargs="+", type=int, default=None)
    parser.add_argument("--shared_expert_task", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--num_vlm_layers", type=int, default=25)
    parser.add_argument("--warmup_rounds", type=int, default=0)
    parser.add_argument("--measure_rounds", type=int, default=20)
    parser.add_argument("--modes", nargs="+", default=["baseline", "task_spec_expert", "tpa_vla"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    summary = summarize(rows)
    out = Path(args.output_dir)
    write_csv(out / "task_switch_overhead_raw.csv", rows)
    write_csv(out / "task_switch_overhead_summary.csv", summary)
    for row in summary:
        print(f"{row['mode']:18s} {row['metric']:28s} mean={row['mean']:.2f} ms p95={row['p95']:.2f} ms")


if __name__ == "__main__":
    main()
