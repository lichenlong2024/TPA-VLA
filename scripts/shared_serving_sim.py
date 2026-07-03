"""Shared resident inference simulator for multi-client requests.

The full paper experiment connects the same resident policy structure to LIBERO
clients. This dependency-light script reproduces the server-side part: one
shared Expert remains resident, task-specific Query modules are selected per
request, and latency/memory metrics are written to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpa_vla.constants import ACTION_DIM, DEFAULT_TASK_TOKENS, PROPRIO_DIM
from tpa_vla.modules import ActionExpert, ProprioProjector, QueryModule, QueryWrappedExpert
from tpa_vla.utils import load_component_state_dict, newest_checkpoint


@dataclass
class Request:
    client_id: int
    task_id: int
    step: int
    created_s: float


def load_query(path: Path, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> QueryModule:
    query = QueryModule(input_dim=hidden_dim)
    query.load_state_dict(load_component_state_dict(path), strict=True)
    return query.to(device=device, dtype=dtype).eval()


def load_expert(path: Path, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> ActionExpert:
    expert = ActionExpert(input_dim=hidden_dim, hidden_dim=hidden_dim)
    expert.load_state_dict(load_component_state_dict(path), strict=True)
    return expert.to(device=device, dtype=dtype).eval()


def load_proprio(path: Path, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> ProprioProjector:
    projector = ProprioProjector(llm_dim=hidden_dim)
    projector.load_state_dict(load_component_state_dict(path), strict=True)
    return projector.to(device=device, dtype=dtype).eval()


def discover_checkpoint_paths(run_root: Path, task_ids: List[int]) -> Dict[int, Dict[str, Path]]:
    paths = {}
    for task_id in task_ids:
        task_dir = run_root / f"task_{task_id}"
        paths[task_id] = {
            "query": newest_checkpoint(task_dir / "query", "query_module--*.pt"),
            "expert": newest_checkpoint(task_dir / "expert", "action_expert--*.pt"),
            "proprio": newest_checkpoint(task_dir / "expert", "proprio_projector--*.pt"),
        }
    return paths


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, help="Root containing task_1/.../task_N checkpoints.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resident_task_count", type=int, default=9)
    parser.add_argument("--requests_per_client", type=int, default=20)
    parser.add_argument("--request_interval_ms", type=float, default=80.0)
    parser.add_argument("--shared_expert_task", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--num_vlm_layers", type=int, default=25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for meaningful serving latency measurements.")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    task_ids = list(range(1, args.resident_task_count + 1))
    paths = discover_checkpoint_paths(Path(args.run_root), task_ids)

    expert = load_expert(paths[args.shared_expert_task]["expert"], args.hidden_dim, device, dtype)
    proprio_projector = load_proprio(paths[args.shared_expert_task]["proprio"], args.hidden_dim, device, dtype)
    queries = {task_id: load_query(paths[task_id]["query"], args.hidden_dim, device, dtype) for task_id in task_ids}
    policies = {task_id: QueryWrappedExpert(queries[task_id], expert).eval() for task_id in task_ids}

    hidden = torch.randn(
        1,
        args.num_vlm_layers,
        DEFAULT_TASK_TOKENS + ACTION_DIM,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    )
    proprio = torch.zeros(1, PROPRIO_DIM, device=device, dtype=dtype)
    requests: "queue.Queue[Optional[Request]]" = queue.Queue()
    rows: List[Dict[str, object]] = []

    def client_loop(client_id: int, task_id: int) -> None:
        for step in range(args.requests_per_client):
            requests.put(Request(client_id=client_id, task_id=task_id, step=step, created_s=time.perf_counter()))
            time.sleep(args.request_interval_ms / 1000.0)

    def server_loop(total_requests: int) -> None:
        for _ in range(total_requests):
            request = requests.get()
            if request is None:
                break
            start = time.perf_counter()
            _ = policies[request.task_id].predict_action(hidden, proprio, proprio_projector)
            torch.cuda.synchronize()
            end = time.perf_counter()
            rows.append(
                {
                    "client_id": request.client_id,
                    "task_id": request.task_id,
                    "step": request.step,
                    "queue_ms": (start - request.created_s) * 1000.0,
                    "service_ms": (end - start) * 1000.0,
                    "end_to_end_ms": (end - request.created_s) * 1000.0,
                    "cuda_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                }
            )

    total = args.resident_task_count * args.requests_per_client
    server = threading.Thread(target=server_loop, args=(total,), daemon=True)
    server.start()
    clients = [
        threading.Thread(target=client_loop, args=(client_id, task_id), daemon=True)
        for client_id, task_id in enumerate(task_ids, start=1)
    ]
    for client in clients:
        client.start()
    for client in clients:
        client.join()
    server.join()

    out = Path(args.output_dir)
    write_csv(out / "request_trace.csv", rows)
    service = [float(row["service_ms"]) for row in rows]
    queue_ms = [float(row["queue_ms"]) for row in rows]
    summary = {
        "resident_task_count": args.resident_task_count,
        "total_requests": len(rows),
        "service_mean_ms": statistics.mean(service),
        "service_p95_ms": sorted(service)[int(0.95 * (len(service) - 1))],
        "queue_mean_ms": statistics.mean(queue_ms),
        "queue_p95_ms": sorted(queue_ms)[int(0.95 * (len(queue_ms) - 1))],
        "cuda_allocated_gb_final": torch.cuda.memory_allocated() / (1024**3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
