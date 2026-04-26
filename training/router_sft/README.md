# Router SFT

This directory isolates Phase 1 router training for `Cortex` without changing the
existing runtime path.

## What it does

1. `collector.py`
   - Runs heuristic `Cortex` episodes with fixed `perception`, `world_modeler`,
     `planner`, and `critic` roles.
   - Wraps the heuristic `ExecutiveRole` as the teacher.
   - Captures one row per Executive invocation.

2. `export_jsonl.py`
   - Splits rows by episode, not by individual row.
   - Produces:
     - `router_sft_train.jsonl`
     - `router_sft_val.jsonl`
     - `router_cls_train.jsonl`
     - `router_cls_val.jsonl`

3. `train.py`
   - Runs standalone LoRA SFT against the chat-style JSONL export.

## Raw row schema

Each raw row contains:

- `episode_id`
- `seed`
- `outer_turn`
- `inner_iteration`
- `input_payload`
- `teacher_decision`
- `route_label`
- `final_action`
- `immediate_reward`
- `return_to_go`
- `episode_total_reward`
- `termination_reason`

`route_label` is one of:

- `act`
- `call_world_modeler`
- `call_planner`
- `call_critic`
- `escalate`
- `wait`
- `stop`

## Collection

```bash
python -m training.router_sft.collector \
  --output outputs/router/router_raw.jsonl \
  --num-episodes 512 \
  --seed-start 0 \
  --budget 30 \
  --num-regions 4 \
  --max-turns 20 \
  --initial-infected 10 \
  --noise-level 0.1
```

## Export

```bash
python -m training.router_sft.export_jsonl \
  --raw outputs/router/router_raw.jsonl \
  --output-dir outputs/router/sft
```

## Train

Install the isolated training dependencies first:

```bash
pip install -r training/router_sft/requirements.txt
```

Then run LoRA SFT:

```bash
python -m training.router_sft.train \
  --train-path outputs/router/sft/router_sft_train.jsonl \
  --val-path outputs/router/sft/router_sft_val.jsonl \
  --model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --output-dir outputs/router/checkpoints/mistral-router
```

## Notes

- This package does not modify `cortex/`, `agents/`, `evaluation/`, or `inference.py`.
- The exported prompt is router-specific and does not rely on the current
  runtime Executive prompt.
- `return_to_go` is currently computed from the outer turn onward. That is
  enough for SFT bootstrap and useful later for preference or RL relabeling.
