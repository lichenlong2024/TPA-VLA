# TPA-VLA

This repository contains the reviewer-facing core implementation for
**TPA-VLA: Two-Phase Adaptation for shared VLA inference**.

TPA-VLA addresses a deployment constraint that appears when multiple robot tasks
share one resident VLA service. A task-specific Expert learned with a plastic
backbone is expensive to keep per task. TPA-VLA instead learns a reusable Action
Expert in Phase I, restores the VLM backbone to its frozen deployed state in
Phase II, and trains only lightweight task Query modules so that the shared
Expert remains compatible with frozen VLM features.

## What Is Included

- `tpa_vla/modules.py`: QueryModule, ActionExpert, ProprioProjector, and the inference wrapper.
- `scripts/train_phase1_expert.py`: trains the shared Expert from adapted-backbone hidden states.
- `scripts/train_phase2_query.py`: trains a task Query with the Expert frozen.
- `scripts/eval_cached_policy.py`: quick validation on cached hidden-state/action data.
- `scripts/task_switch_overhead_microbenchmark.py`: reproduces the task-switching cost measurement logic.
- `scripts/shared_serving_sim.py`: simulates concurrent client requests to one resident shared service.
- `docs/reproducibility.md`: checkpoint layout, cache format, and example commands.

## What Is Not Included

This minimal release intentionally excludes raw robot/LIBERO data, checkpoints,
videos, private paths, and unrelated exploratory code. It is designed to make the
paper method auditable without publishing non-paper experiments or large assets.

## Install

```bash
git clone https://github.com/lichenlong2024/TPA-VLA.git
cd TPA-VLA
pip install -e .
```

The public scripts use cached VLM hidden states so they can be inspected without
reproducing the full OpenVLA training stack. To run full LIBERO rollouts, connect
`QueryWrappedExpert` to the standard OpenVLA/OpenVLA-OFT LIBERO evaluation loop.

## Two-Phase Training

Phase I trains the ActionExpert using hidden states from the temporarily adapted
VLM:

```bash
python scripts/train_phase1_expert.py \
  --train_cache /path/to/phase1_adapted_hidden_states.pt \
  --output_dir /path/to/run_root/task_1/expert
```

Phase II restores the frozen VLM representation and trains only a QueryModule
through the fixed Expert:

```bash
python scripts/train_phase2_query.py \
  --train_cache /path/to/task_i_frozen_hidden_states.pt \
  --expert_checkpoint /path/to/run_root/task_1/expert/action_expert--step10000.pt \
  --proprio_checkpoint /path/to/run_root/task_1/expert/proprio_projector--step10000.pt \
  --output_dir /path/to/run_root/task_i/query
```

The cache format is documented in `docs/reproducibility.md`.

## Shared-Serving Experiments

Task-switching overhead:

```bash
python scripts/task_switch_overhead_microbenchmark.py \
  --run_root /path/to/run_root \
  --output_dir /path/to/switch_cost
```

Concurrent request simulation:

```bash
python scripts/shared_serving_sim.py \
  --run_root /path/to/run_root \
  --resident_task_count 9 \
  --output_dir /path/to/shared_serving
```

Both scripts isolate server-side behavior. They do not claim to measure
simulator stepping, robot communication, or complete end-to-end control latency.

## Citation

If this code is useful, please cite the accompanying TPA-VLA paper.
