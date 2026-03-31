# Input Distribution Approximation (L2L)

A training pipeline for learning to approximate input distributions between frozen Qwen2.5-7B-Instruct model instances using a trainable intermediary model.

## Architecture

Three Qwen2.5-7B-Instruct model roles form a differentiable chain:

```
Prompt -> [Model A (frozen)] -> softmax(logits / temp)
       -> prepend <|im_start|>assistant, append <|im_start|>user
       -> [Middle Model (trainable)] -> embed_tokens -> layers -> lm_head (frozen) -> softmax
       -> [Model B (frozen)] -> embed_tokens -> layers -> logits
       -> CE loss vs Model A targets (shifted autoregressive)
```

- **Model A** (frozen): Generates soft probability sequences from prompts via autoregressive generation
- **Middle Model** (partially trainable): Qwen2.5-7B copy with reinitialized `embed_tokens` and last 8 transformer layers (20-27). The `lm_head` remains frozen
- **Model B** (frozen): Receives the middle model's output mapped back into embedding space and reconstructs Model A's probability distributions

Model A and B share one frozen model instance to save memory.

## Training

Two-stage training process:

**Stage 1** (1 epoch) — Train only the reinitialized parameters:
- `embed_tokens` weight matrix
- Transformer layers 20-27

**Stage 2** (configurable epochs) — Full finetune of the middle model:
- Two-tier learning rate: lower for pretrained layers (0-19), higher for reinitialized layers
- `lm_head` remains frozen throughout
- Model A's softmax temperature decays (cosine schedule, 1.0 -> 0.05) so the middle model gradually learns to interpret discrete token distributions

## Files

| File | Description |
|------|-------------|
| `l2l_pipeline.py` | Main training pipeline (Model A -> Middle -> Model B) |
| `qwen_logit_inference.py` | Batch logit extraction from Qwen2.5-7B-Instruct |
| `reinit_qwen_layers.py` | Reinitialize last N transformer layers of a Qwen model |
| `merge_datasets.py` | Merge ShareGPT and harmful-dataset into `merged_prompts.jsonl` |
| `data/merged_prompts.jsonl` | 91,342 prompts (ShareGPT + LLM-LAT/harmful-dataset) |

## Usage

### Training

```bash
python l2l_pipeline.py \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-new-tokens 256 \
  --stage1-epochs 1 \
  --stage2-epochs 3
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 2 | Prompts per forward pass |
| `--gradient-accumulation-steps` | 8 | Steps before optimizer update (effective batch = 16) |
| `--max-new-tokens` | 256 | Max tokens Model A generates per prompt |
| `--stage1-lr` | 2e-4 | Learning rate for stage 1 |
| `--stage2-lr-high` | 1e-4 | LR for reinitialized layers in stage 2 |
| `--stage2-lr-low` | 1e-5 | LR for pretrained layers in stage 2 |
| `--temp-end` | 0.05 | Final temperature in stage 2 |
| `--temp-schedule` | cosine | Temperature decay schedule (`cosine` or `linear`) |
| `--checkpoint-dir` | `checkpoints/l2l/` | Where to save checkpoints |

### Logit Extraction

```bash
python qwen_logit_inference.py --batch-size 16 --output /tmp/qwen_logits.buf
```

Streams the full 152,064-dim logit vector (float16) for each prompt to a binary buffer file.

### Layer Reinitialization

```bash
python reinit_qwen_layers.py --num-reinit-layers 8 --output-dir models/qwen2.5-7b-reinit-last8
```

### Dataset Preparation

```bash
python merge_datasets.py --seed 42 --output data/merged_prompts.jsonl
```

## Requirements

- Python 3.10+
- PyTorch 2.10+
- Transformers >= 4.43.0
- Accelerate >= 0.27.0
- Qwen2.5-7B-Instruct model weights

```bash
pip install -r requirements.txt
```
