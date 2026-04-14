"""
L2L Sampling / Test Script.

Tests the trained middle model by running the full pipeline on a single prompt
with all weights frozen, decoding outputs at each stage:

  Prompt -> [Model A (base Qwen, GPU 3)] -> generate -> raw logits
         -> build_soft_sequence -> prob_vectors
         -> [Middle Model (checkpoint, GPU 4)] -> forward -> argmax -> decode
         -> [Model B (base Qwen, GPU 3)] -> generate from middle tokens -> decode

Uses GPUs 3-4 to avoid interfering with training on GPUs 0-2.

Usage:
  python l2l_sample.py
"""

import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
# CHECKPOINT_PATH = os.path.join(
#     os.path.dirname(__file__), "checkpoints", "l2l_fast", "stage1_final.pt",
# )
CHECKPOINT_PATH = "checkpoints/l2l_stable_fast_old6/stage1_final.pt"

IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
USER_ID = 872
VOCAB_SIZE = 152064

GPU_BASE = 3
GPU_MIDDLE = 4

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
PROMPT = "Explain what a neural network is in one paragraph."
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0  # stage 1 trained at 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_attn_impl():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_base_model(gpu, dtype, attn_impl):
    """Load frozen base Qwen on the given GPU."""
    print(f"  Loading base model on cuda:{gpu}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype,
        device_map={"": gpu},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_middle_from_checkpoint(gpu, ckpt_path, dtype, attn_impl):
    """Load base Qwen on GPU, then restore middle model weights from checkpoint."""
    print(f"  Loading middle model on cuda:{gpu}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype,
        device_map={"": gpu},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    print(f"  Restoring checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["middle_model_state_dict"])
    stage = ckpt.get("stage", "?")
    step = ckpt.get("step", "?")
    print(f"  Restored (stage {stage}, step {step})")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_prompt(tokenizer, prompt, device):
    """Tokenize a single prompt using the Qwen chat template."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def generate_model_a(model, tokenizer, input_ids, attention_mask):
    """
    Model A: autoregressive generation with logit capture.
    Returns (raw_logits, gen_lengths, generated_text).
    """
    gen_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
    )

    raw_logits = torch.stack(gen_output.logits, dim=1)  # (1, gen, V)

    prompt_len = input_ids.shape[1]
    gen_tokens = gen_output.sequences[:, prompt_len:]
    is_eos = gen_tokens == IM_END_ID
    has_eos = is_eos.any(dim=1)
    first_eos = is_eos.int().argmax(dim=1)
    gen_lengths = torch.where(
        has_eos, first_eos,
        torch.full_like(first_eos, gen_tokens.shape[1]),
    ).clamp(min=1)

    # Decode generated text (up to first EOS)
    gen_len = gen_lengths[0].item()
    gen_text = tokenizer.decode(
        gen_tokens[0, :gen_len], skip_special_tokens=True,
    )

    return raw_logits, gen_lengths, gen_text


def build_soft_sequence(raw_logits, gen_lengths, temperature, dtype):
    """
    Build soft probability sequence with <|im_start|>assistant prefix
    and <|im_start|>user suffix.
    Returns (prob_vectors, mask).
    """
    device = raw_logits.device
    temp = max(temperature, 1e-6)
    probs = F.softmax(raw_logits / temp, dim=-1)

    max_gen = gen_lengths.max().item()
    max_total = max_gen + 4
    B = raw_logits.shape[0]

    output = torch.zeros(B, max_total, VOCAB_SIZE, device=device, dtype=dtype)

    # Prefix
    output[:, 0, IM_START_ID] = 1.0
    output[:, 1, ASSISTANT_ID] = 1.0

    # Prob vectors (masked per item)
    gen_idx = torch.arange(max_gen, device=device).unsqueeze(0)
    gen_valid = gen_idx < gen_lengths.unsqueeze(1)
    output[:, 2:2 + max_gen] = probs[:, :max_gen] * gen_valid.unsqueeze(-1)

    # Suffix
    batch_idx = torch.arange(B, device=device)
    suffix_start = gen_lengths + 2
    output[batch_idx, suffix_start, IM_START_ID] = 1.0
    output[batch_idx, suffix_start + 1, USER_ID] = 1.0

    # Mask
    total_len = suffix_start + 2
    pos_idx = torch.arange(max_total, device=device).unsqueeze(0)
    mask = (pos_idx < total_len.unsqueeze(1)).to(dtype)

    output = output.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
    return output, mask


def run_middle_model(model, tokenizer, prob_vectors, mask, device):
    """
    Forward pass through the middle model.
    Returns (argmax_ids_valid, decoded_text).
    """
    # Route through CPU to avoid P2P GPU transfer issues
    prob_mid = prob_vectors.cpu().to(device)
    mask_mid = mask.cpu().to(device)

    soft_embeds = prob_mid @ model.model.embed_tokens.weight

    middle_out = model(
        inputs_embeds=soft_embeds,
        attention_mask=mask_mid,
        use_cache=False,
    )

    middle_logits = middle_out.logits.float().clamp(-65504, 65504)
    argmax_ids = middle_logits.argmax(dim=-1)  # (1, S)

    # Extract valid positions
    valid_mask = mask_mid[0].bool()
    valid_ids = argmax_ids[0, valid_mask]

    decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)
    return valid_ids, decoded


def generate_model_b(model, tokenizer, middle_ids, device):
    """
    Model B: generate continuation from the middle model's decoded tokens.
    """
    input_ids = middle_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    gen_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,
        do_sample=False,
    )

    # Decode only the newly generated tokens
    new_tokens = gen_output[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return gen_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    dtype = torch.bfloat16
    attn_impl = detect_attn_impl()
    device_base = torch.device(f"cuda:{GPU_BASE}")
    device_mid = torch.device(f"cuda:{GPU_MIDDLE}")

    print("=== Configuration ===")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Attention: {attn_impl}")
    print(f"  GPUs: base={GPU_BASE}, middle={GPU_MIDDLE}")

    # -- Load models --------------------------------------------------------
    print("\n=== Loading Models ===")
    tokenizer = load_tokenizer()
    base_model = load_base_model(GPU_BASE, dtype, attn_impl)
    middle_model = load_middle_from_checkpoint(
        GPU_MIDDLE, CHECKPOINT_PATH, dtype, attn_impl,
    )

    # GPU report
    for i in [GPU_BASE, GPU_MIDDLE]:
        mem = torch.cuda.memory_allocated(i) / (1024 ** 3)
        print(f"  GPU {i}: {mem:.1f} GB allocated")

    # -- Tokenize prompt ----------------------------------------------------
    print(f"\n=== Prompt ===")
    print(f"  {PROMPT}")
    input_ids, attention_mask = tokenize_prompt(tokenizer, PROMPT, device_base)
    print(f"  Tokenized: {input_ids.shape[1]} tokens")

    # -- Model A: generate --------------------------------------------------
    print(f"\n=== Model A Output (Base Qwen, greedy) ===")
    raw_logits, gen_lengths, gen_text = generate_model_a(
        base_model, tokenizer, input_ids, attention_mask,
    )
    print(gen_text)
    print(f"\n  [Generated {gen_lengths[0].item()} tokens, "
          f"logits shape: {raw_logits.shape}]")

    # -- Build soft sequence ------------------------------------------------
    prob_vectors, mask = build_soft_sequence(
        raw_logits, gen_lengths, TEMPERATURE, dtype,
    )
    seq_len = int(mask[0].sum().item())
    print(f"  [Soft sequence: {seq_len} positions "
          f"(2 prefix + {gen_lengths[0].item()} gen + 2 suffix)]")

    # -- Middle model: forward + decode -------------------------------------
    print(f"\n=== Middle Model Output (Decoded) ===")
    middle_ids, middle_text = run_middle_model(
        middle_model, tokenizer, prob_vectors, mask, device_mid,
    )
    print(middle_text)
    print(f"\n  [{middle_ids.shape[0]} tokens decoded from middle model]")

    # -- Model B: generate from middle tokens -------------------------------
    print(f"\n=== Model B Output (Reconstruction from Middle) ===")
    if middle_ids.shape[0] == 0:
        print("  [Skipped — no valid tokens from middle model]")
    else:
        model_b_text = generate_model_b(
            base_model, tokenizer, middle_ids, device_base,
        )
        if model_b_text:
            print(model_b_text)
        else:
            print("  [No new tokens generated]")
        print(f"\n  [Generated from {middle_ids.shape[0]} middle model tokens]")

    print("\nDone.")


if __name__ == "__main__":
    main()
