# Input Distribution Approximation via Inverse Mapping

Fine-tunes LLaMA-3.1-8B to predict human prompts from model responses — learning the inverse mapping $F^{-1}: Y \to X$ to approximate the input distribution that produces a given output.

## Motivation

Standard distillation handles $F: X \to Y$ distilling into $F': X \to Y$. This project inverts that: given a model response, predict the human prompt that produced it. This is useful for interpretability research — understanding what inputs lead to specific model behaviors.

## Architecture

The inverse model is a full LLaMA-3.1-8B fine-tuned with a two-stage training pipeline:

**Stage 1 — Warmup (DDP):** Reinitializes and trains only the final 4 transformer layers (28–31) while freezing all pretrained weights. Runs 1 epoch across 5 GPUs with DDP.

**Stage 2 — Full Training (FSDP):** Unfreezes all parameters and trains with differential learning rates — lower LR (2e-5) for pretrained weights, higher LR (2e-4) for reinitialized layers. Runs 3 epochs across 5 GPUs with FSDP (FULL_SHARD).

### Sequence Format

```
[BOS] gpt_response [SEP] human_prompt [EOS]
```

- GPT response tokens are masked (`-100`) — they serve as context only
- Human prompt tokens are the training targets
- `[SEP]` = `<|reserved_special_token_0|>` (token ID 128002)

## Data

Training uses [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) with GPT-4 responses replaced by LLaMA-3.1-8B generations (86,394 multi-turn conversations). Response generation scripts are included (`generate_llama_responses.py`, `generate_llama_responses_fast.py`).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires 5 GPUs with CUDA 12.8+. The base model should be at the path specified in `training/config.py`.

## Training

**Stage 1 (DDP warmup):**
```bash
accelerate launch --config_file accelerate_config_stage1.yaml training/train.py
```

**Stage 2 (FSDP full training):**
```bash
accelerate launch --config_file accelerate_config.yaml training/train.py
```

After Stage 2, convert the FSDP sharded checkpoint to HuggingFace format:
```bash
python convert_sharded_checkpoint.py
```

## Evaluation

```bash
python evaluate.py
```

Computes validation loss, perplexity, BLEU, and ROUGE-L on held-out conversations.

## Interactive Testing

Round-trip pipeline: GPT response → predicted human prompt → LLaMA response.

```bash
# Interactive mode
python test_pipeline.py

# Single prompt
python test_pipeline.py --prompt "Here is a GPT response to process..."

# Custom devices
python test_pipeline.py --inverse-device cuda:0 --base-device cuda:1
```

## Project Structure

```
├── training/
│   ├── config.py          # TrainingConfig dataclass
│   ├── data.py            # FlippedConversationDataset, collation, dataloaders
│   ├── model_setup.py     # Model loading, layer reinitialization, optimizer groups
│   ├── train.py           # Two-stage training loop
│   └── utils.py           # Metrics, checkpointing, evaluation
├── evaluate.py            # Evaluation script (loss, perplexity, BLEU, ROUGE-L)
├── test_pipeline.py       # Interactive round-trip testing
├── generate_llama_responses.py       # Single-sequence response generation
├── generate_llama_responses_fast.py  # Batched response generation
├── convert_sharded_checkpoint.py     # FSDP → HuggingFace checkpoint conversion
├── accelerate_config.yaml            # Stage 2 FSDP config (5 GPUs)
├── accelerate_config_stage1.yaml     # Stage 1 DDP config (5 GPUs)
└── caveats.md             # Notes on domain mismatch and design decisions
```

## Results

| Metric | Value |
|--------|-------|
| Val Loss | 1.88 |
| Val Perplexity | 6.57 |
| Avg BLEU | 0.265 |
| Avg ROUGE-L | 0.317 |
