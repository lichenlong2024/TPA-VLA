# Reproducibility Notes

This repository exposes the TPA-VLA-specific code used by the paper:

- `QueryModule`: the lightweight task module switched at inference.
- `ActionExpert`: the continuous action-chunk decoder shared across tasks.
- `train_phase1_expert.py`: trains the Expert from temporarily adapted VLM hidden states.
- `train_phase2_query.py`: freezes the Expert and trains only the task QueryModule.
- `task_switch_overhead_microbenchmark.py`: isolates server-side task-switching overhead.
- `shared_serving_sim.py`: simulates concurrent clients against one resident shared service.

The paper experiments used LIBERO and real-robot data. Raw datasets, checkpoints,
videos, and machine-local paths are intentionally not included in this reviewer
code release.

## Hidden-State Cache Format

The public training scripts expect a `.pt`, `.pth`, or `.npz` file with:

```python
{
    "hidden_states": Tensor[N, layers, tokens, hidden_dim],
    "proprio": Tensor[N, proprio_dim],
    "actions": Tensor[N, chunk, action_dim],
}
```

For Phase I, hidden states should come from the temporarily adapted VLM. For
Phase II, hidden states should come from the restored frozen VLM.

## Default Hyperparameters

The defaults match the controlled paper setting unless changed at the command
line:

- batch size: 16
- learning rate: 1e-4
- gradient clipping: 1.0
- action chunk: 8
- action dimension: 7
- proprio dimension: 8
- Query layers: 3
- Query attention heads: 8
- training steps per task: 10,000

## Example Commands

```bash
python scripts/train_phase1_expert.py \
  --train_cache /path/to/phase1_adapted_hidden_states.pt \
  --output_dir /path/to/run_root/task_1/expert

python scripts/train_phase2_query.py \
  --train_cache /path/to/task_2_frozen_hidden_states.pt \
  --expert_checkpoint /path/to/run_root/task_1/expert/action_expert--step10000.pt \
  --proprio_checkpoint /path/to/run_root/task_1/expert/proprio_projector--step10000.pt \
  --output_dir /path/to/run_root/task_2/query

python scripts/task_switch_overhead_microbenchmark.py \
  --run_root /path/to/run_root \
  --output_dir /path/to/switch_cost

python scripts/shared_serving_sim.py \
  --run_root /path/to/run_root \
  --resident_task_count 9 \
  --output_dir /path/to/shared_serving
```
