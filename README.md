# TPA-VLA

Reviewer-facing code for **TPA-VLA: Two-Phase Adaptation for shared VLA
inference**.

TPA-VLA trains a reusable continuous Action Expert in Phase I with temporarily
adapted VLM representations. In Phase II, the deployed VLM is restored to its
frozen state, the Expert is fixed, and only a lightweight task Query Module is
trained. At inference time, multiple tasks share the same frozen VLM and the
same Expert; serving a new task only switches its Query.

## Repository Contents

- `tpa_vla/modules.py`: `QueryModule`, `ActionExpert`, `ProprioProjector`, and `QueryWrappedExpert`.
- `scripts/extract_hidden_cache.py`: exports adapted/frozen VLM hidden-state caches from a JSONL manifest.
- `scripts/train_phase1_expert.py`: Phase-I Expert training.
- `scripts/train_phase2_query.py`: Phase-II Query-only recovery.
- `scripts/run_tpa_pipeline.py`: one-command Phase-I/Phase-II/eval pipeline from YAML.
- `scripts/run_expert_source_grid.py`: Expert-source cross-validation grid.
- `scripts/task_switch_overhead_microbenchmark.py`: low-memory task-switching overhead benchmark.
- `scripts/shared_serving_sim.py`: concurrent request simulation with one resident shared Expert.

No checkpoints, raw datasets, or videos are included.

## Install

```bash
git clone https://github.com/lichenlong2024/TPA-VLA.git
cd TPA-VLA
pip install -e .
```

The smoke test can also be run directly from the repository root without
editable installation because each script adds the repository root to
`PYTHONPATH`.

For hidden-cache extraction from Hugging Face VLMs, also install the model's
usual dependencies, such as `transformers`, `pillow`, and optionally `peft`.

## 0. Quick Smoke Test

This synthetic test does not evaluate robot performance. It verifies that the
released code can train an Expert, train a Query through the frozen Expert, and
evaluate the composed policy.

This path has been smoke-tested with Python 3.8 and PyTorch 1.11. Newer PyTorch
versions should also work.

```bash
python scripts/make_toy_cache.py --output runs/toy/phase1.pt --hidden_dim 32 --num_layers 5 --num_samples 64
python scripts/make_toy_cache.py --output runs/toy/phase2.pt --hidden_dim 32 --num_layers 5 --num_samples 64 --seed 1
python scripts/make_toy_cache.py --output runs/toy/eval.pt --hidden_dim 32 --num_layers 5 --num_samples 32 --seed 2

python scripts/train_phase1_expert.py \
  --train_cache runs/toy/phase1.pt \
  --output_dir runs/toy/shared_expert \
  --hidden_dim 32 \
  --num_blocks 4 \
  --batch_size 8 \
  --max_steps 5

python scripts/train_phase2_query.py \
  --train_cache runs/toy/phase2.pt \
  --expert_checkpoint runs/toy/shared_expert/action_expert--step5.pt \
  --proprio_checkpoint runs/toy/shared_expert/proprio_projector--step5.pt \
  --output_dir runs/toy/task_query \
  --hidden_dim 32 \
  --expert_blocks 4 \
  --query_layers 1 \
  --batch_size 8 \
  --max_steps 5

python scripts/eval_cached_policy.py \
  --eval_cache runs/toy/eval.pt \
  --expert_checkpoint runs/toy/shared_expert/action_expert--step5.pt \
  --proprio_checkpoint runs/toy/shared_expert/proprio_projector--step5.pt \
  --query_checkpoint runs/toy/task_query/query_module--final.pt \
  --hidden_dim 32 \
  --expert_blocks 4 \
  --query_layers 1
```

## 1. Hidden-State Cache Extraction

The public training scripts use cached VLM hidden states. This keeps the
TPA-VLA-specific code independent of a particular private training launcher and
makes the Phase-I/Phase-II distinction explicit.

Create a JSONL manifest:

```json
{"image": "frames/000001.png", "instruction": "put the mug in the microwave", "proprio": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]}
```

Each row should provide `image`, `instruction`, `proprio`, and an action chunk.
For LIBERO, the image/instruction/proprio/action fields can be exported from the
standard rollout or RLDS loader.

Phase-I adapted-backbone cache:

```bash
python scripts/extract_hidden_cache.py \
  --manifest /path/to/train_manifest.jsonl \
  --image_root /path/to/images \
  --model_path /path/to/base_or_phase1_vlm \
  --lora_adapter_path /path/to/phase1_lora_adapter \
  --output runs/cache/task1_phase1_adapted.pt
```

Phase-II frozen-backbone cache:

```bash
python scripts/extract_hidden_cache.py \
  --manifest /path/to/train_manifest.jsonl \
  --image_root /path/to/images \
  --model_path /path/to/frozen_vlm \
  --output runs/cache/task1_phase2_frozen.pt
```

If your VLM returns hidden states through a model-specific method rather than
`output_hidden_states=True`, modify only `extract_hidden_cache.py`; the rest of
the TPA-VLA pipeline remains unchanged.

## 2. Phase-I and Phase-II Training

Train the shared Expert:

```bash
python scripts/train_phase1_expert.py \
  --train_cache runs/cache/task1_phase1_adapted.pt \
  --output_dir runs/tpa/shared_expert \
  --hidden_dim 896 \
  --batch_size 16 \
  --max_steps 10000 \
  --learning_rate 1e-4
```

Train a task Query with the Expert fixed:

```bash
python scripts/train_phase2_query.py \
  --train_cache runs/cache/task2_phase2_frozen.pt \
  --expert_checkpoint runs/tpa/shared_expert/action_expert--step10000.pt \
  --proprio_checkpoint runs/tpa/shared_expert/proprio_projector--step10000.pt \
  --output_dir runs/tpa/task2/query \
  --hidden_dim 896 \
  --query_layers 3 \
  --query_heads 8 \
  --batch_size 16 \
  --max_steps 10000 \
  --learning_rate 1e-4
```

Or use the YAML pipeline:

```bash
python scripts/run_tpa_pipeline.py --config configs/pipeline_example.yaml
```

## 3. Expert-Source Cross-Validation

To reproduce the Expert-source sensitivity study, list the five source/target
tasks in `configs/expert_source_grid_example.yaml`, then run:

```bash
python scripts/run_expert_source_grid.py \
  --config configs/expert_source_grid_example.yaml \
  --output_csv runs/expert_source_grid/grid_summary.csv
```

This script trains one Phase-I Expert per source task and one Phase-II Query per
source-target pair. It reports cached validation L1 loss. To convert each cell
to LIBERO success rate, evaluate the saved `E_<source>__T_<target>/query`
checkpoint with the same source Expert in your LIBERO rollout loop.

## 4. LIBERO Rollout Evaluation

The paper uses standard LIBERO rollouts with the following policy composition:

```python
from tpa_vla.modules import ActionExpert, ProprioProjector, QueryModule, QueryWrappedExpert

policy = QueryWrappedExpert(query_module, shared_expert)
actions = policy.predict_action(vlm_hidden_states, proprio, proprio_projector)
```

In an OpenVLA/OpenVLA-OFT LIBERO evaluator, replace the action head with
`QueryWrappedExpert(query, expert)` after the VLM hidden states are produced.
The checkpoint layout expected by the scripts is shown in
`configs/example_paths.yaml`.

## 5. Task-Switching Cost

The paper's switch-cost microbenchmark isolates module preparation cost under a
low-memory dynamic-serving setting. It excludes simulator stepping, network
communication, image preprocessing, and full VLM forward cost.

```bash
python scripts/task_switch_overhead_microbenchmark.py \
  --run_root /path/to/run_root \
  --output_dir runs/switch_cost \
  --num_tasks 5
```

Expected checkpoint layout:

```text
run_root/
  task_1/
    expert/action_expert--step10000.pt
    expert/proprio_projector--step10000.pt
    query/query_module--final.pt
    adapter/adapter_model.pt       # optional for baseline mode
```

## 6. Multi-Client Shared Serving

The concurrent request simulator keeps one Expert resident, loads all task
Queries, and sends multiple client request streams through the shared service:

```bash
python scripts/shared_serving_sim.py \
  --run_root /path/to/run_root \
  --resident_task_count 9 \
  --requests_per_client 20 \
  --output_dir runs/shared_serving
```

It writes `request_trace.csv` and `summary.json` with queue, service, and
end-to-end server-side latency metrics.

## Notes

The public code is organized around cached hidden states because this is the
cleanest way to expose the method-specific contribution without bundling large
datasets, checkpoints, or a private training environment. The same modules can
be inserted into a full VLA training/evaluation stack by replacing the cached
hidden states with live VLM forward outputs.
