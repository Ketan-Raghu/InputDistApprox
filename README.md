# Input Distribution Approximation — T2T Pipeline

Train a partially-reinitialized Qwen2.5-7B-Instruct to **predict the original input prompt given a single model output**. Each prompt generates 5 independent outputs (one per GPU via different seeds), and each (output, prompt) pair is a separate training sample.

## Running the Full Pipeline

```bash
# 1. Merge datasets (if data/merged_prompts.jsonl doesn't exist)
python merge_datasets.py

# 2. Validate sampling diversity (optional, requires 1 GPU)
pytest test_t2t_sampling_diversity.py -v

# 3. Generate training pairs (5 GPUs, ~456K pairs)
python generate_t2t_outputs.py --num-gpus 5

# 4. Train (DDP across 5 GPUs)
torchrun --nproc_per_node=5 train_t2t.py
```

## Architecture

```
merged_prompts.jsonl (91K prompts)
        |
        v
  generate_t2t_outputs.py
  5 GPUs, each with seed=42+gpu_id
  Each generates 1 output per prompt (temp=0.9)
        |
        v
  data/t2t_training_pairs.jsonl (~456K rows)
  Format: {"prompt": "...", "output": "...", "gpu_id": N}
        |
        v
  train_t2t.py (torchrun --nproc=5)
  Model: Qwen2.5-7B-Instruct, layers 24-27 reinitialized
  Input:  system + user:{model_output} + assistant:{original_prompt}
  Loss:   CE on assistant tokens only
        |
        v
  checkpoints/t2t/final_model/
```

## Training Stages

**Epoch 0 — Stabilization:** Only reinit layers (24-27) trainable, `lr=2e-4`

**Epochs 1-3 — Full fine-tuning:** All params trainable, 2-tier LR:
- Reinit layers (24-27): `lr=1e-4`
- Pretrained layers: `lr=1e-5`

## Files

| File | Description |
|------|-------------|
| `generate_t2t_outputs.py` | 5-GPU parallel output generation with seed-based diversity |
| `train_t2t.py` | DDP training: stabilization epoch then full finetune |
| `test_t2t_sampling_diversity.py` | Pytest: validates seed-based sampling produces diverse outputs |
| `reinit_qwen_layers.py` | `reinit_module()` — resets layer params to normal(0, 0.02) |
| `merge_datasets.py` | Merge ShareGPT + harmful-dataset into `merged_prompts.jsonl` |

## Key Arguments

### Generation (`generate_t2t_outputs.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 64 | Prompts per forward pass per GPU |
| `--max-new-tokens` | 512 | Max tokens generated per output |
| `--temperature` | 0.9 | Sampling temperature |
| `--num-gpus` | 5 | Number of GPUs for parallel generation |
| `--resume` | off | Resume from existing per-GPU output files |
| `--merge-only` | off | Skip generation, just merge existing files |

### Training (`train_t2t.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 2 | Per-GPU batch size |
| `--gradient-accumulation-steps` | 4 | Accumulation steps (effective batch = 2 x 4 x 5 = 40) |
| `--stabilization-lr` | 2e-4 | LR for stabilization epoch |
| `--lr-high` | 1e-4 | LR for reinit layers in full finetune |
| `--lr-low` | 1e-5 | LR for pretrained layers in full finetune |
| `--max-seq-len` | 2048 | Max total sequence length |
| `--resume-from` | none | Path to checkpoint to resume from |

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Transformers >= 4.43.0
- 5 GPUs with >= 96 GB each (stabilization ~30 GB/GPU, full training ~95 GB/GPU)
- Qwen2.5-7B-Instruct model weights

```bash
pip install -r requirements.txt
```
