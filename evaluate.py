"""Evaluate the inverse mapping model on the validation split.

Computes:
  - Validation loss and perplexity (teacher-forced)
  - Per-example generation: given GPT response, predict the human prompt
  - BLEU and ROUGE-L scores against ground truth prompts

Usage:
    python evaluate.py [--model_dir checkpoints/inverse_mapping/final_model]
    python evaluate.py --num_generate 50 --max_new_tokens 256
"""

import argparse
import json
import math
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.config import TrainingConfig
from training.data import FlippedConversationDataset, collate_fn, build_datasets


def compute_val_loss(model, dataloader, device) -> dict:
    """Compute average loss and perplexity over the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing val loss"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            # Count target tokens (non -100) for weighted averaging
            target_mask = batch["labels"] != -100
            num_targets = target_mask.sum().item()

            total_loss += outputs.loss.item() * num_targets
            total_tokens += num_targets
            total_examples += batch["input_ids"].size(0)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {
        "val_loss": avg_loss,
        "val_ppl": ppl,
        "num_examples": total_examples,
        "num_target_tokens": total_tokens,
    }


def extract_generation_pairs(dataset, config, tokenizer, max_pairs=None):
    """Extract (gpt_response, human_prompt) pairs from the dataset for generation eval.

    For multi-turn conversations, extracts the first turn only (simplest evaluation).
    """
    pairs = []
    for idx in range(len(dataset)):
        example = dataset[idx]
        input_ids = example["input_ids"].tolist()
        labels = example["labels"].tolist()

        # Find separator positions to identify turn boundaries
        sep_positions = [
            i for i, tok in enumerate(input_ids)
            if tok == config.separator_token_id
        ]
        if not sep_positions:
            continue

        # First turn: [BOS] gpt_response [SEP] human_prompt [SEP or EOS]
        # The GPT response context is everything up to and including the first separator
        first_sep = sep_positions[0]
        context_ids = input_ids[: first_sep + 1]  # [BOS] gpt_response [SEP]

        # The human prompt is from after the first separator to the next separator or EOS
        if len(sep_positions) > 1:
            end = sep_positions[1]
        else:
            # Single turn — prompt goes to EOS
            end = len(input_ids) - 1 if input_ids[-1] == config.eos_token_id else len(input_ids)

        target_ids = input_ids[first_sep + 1: end]

        # Decode
        gpt_response = tokenizer.decode(input_ids[1:first_sep], skip_special_tokens=False)
        human_prompt = tokenizer.decode(target_ids, skip_special_tokens=False)

        if gpt_response.strip() and human_prompt.strip():
            pairs.append({
                "context_ids": context_ids,
                "gpt_response": gpt_response.strip(),
                "human_prompt": human_prompt.strip(),
            })

        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def generate_predictions(model, tokenizer, pairs, config, device, max_new_tokens=256):
    """Generate human prompt predictions given GPT response context."""
    model.eval()
    results = []

    for pair in tqdm(pairs, desc="Generating predictions"):
        input_ids = torch.tensor([pair["context_ids"]], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=[config.eos_token_id, config.separator_token_id],
                pad_token_id=config.pad_token_id,
            )

        # Extract only the generated portion
        generated_ids = output_ids[0, input_ids.size(1):]
        # Trim EOS/SEP if present
        gen_list = generated_ids.tolist()
        for stop_id in [config.eos_token_id, config.separator_token_id]:
            if stop_id in gen_list:
                gen_list = gen_list[:gen_list.index(stop_id)]

        predicted = tokenizer.decode(gen_list, skip_special_tokens=False).strip()
        results.append({
            "gpt_response": pair["gpt_response"],
            "human_prompt": pair["human_prompt"],
            "predicted_prompt": predicted,
        })

    return results


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU-4 with smoothing."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Collect n-gram counts
    scores = []
    for n in range(1, 5):
        ref_ngrams = defaultdict(int)
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams[tuple(ref_tokens[i:i + n])] += 1

        hyp_ngrams = defaultdict(int)
        for i in range(len(hyp_tokens) - n + 1):
            hyp_ngrams[tuple(hyp_tokens[i:i + n])] += 1

        clipped = sum(
            min(count, ref_ngrams.get(ng, 0))
            for ng, count in hyp_ngrams.items()
        )
        total = max(sum(hyp_ngrams.values()), 1)

        # Add-1 smoothing for zero counts
        scores.append((clipped + 1) / (total + 1))

    # Geometric mean
    log_avg = sum(math.log(s) for s in scores) / 4

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    return bp * math.exp(log_avg)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS via DP
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description="Evaluate inverse mapping model")
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Path to HuggingFace model directory (default: <output_dir>/final_model)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for loss computation",
    )
    parser.add_argument(
        "--num_generate", type=int, default=20,
        help="Number of examples to generate predictions for",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-select GPU)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    config = TrainingConfig()
    model_dir = args.model_dir or str(Path(config.output_dir) / "final_model")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {model_dir}")
    print(f"Device: {device}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)

    # Rebuild the same val split used during training
    print("Loading validation data...")
    _, val_dataset = build_datasets(config, tokenizer)
    print(f"Validation examples: {len(val_dataset)}")

    # 1. Compute validation loss/perplexity
    collate = partial(collate_fn, pad_token_id=config.pad_token_id)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    print("\n--- Validation Loss ---")
    loss_metrics = compute_val_loss(model, val_loader, device)
    print(f"  Loss:          {loss_metrics['val_loss']:.4f}")
    print(f"  Perplexity:    {loss_metrics['val_ppl']:.2f}")
    print(f"  Examples:      {loss_metrics['num_examples']}")
    print(f"  Target tokens: {loss_metrics['num_target_tokens']}")

    # 2. Generate predictions
    if args.num_generate > 0:
        print(f"\n--- Generation ({args.num_generate} examples) ---")
        pairs = extract_generation_pairs(
            val_dataset, config, tokenizer, max_pairs=args.num_generate
        )
        results = generate_predictions(
            model, tokenizer, pairs, config, device,
            max_new_tokens=args.max_new_tokens,
        )

        # Compute metrics
        bleu_scores = []
        rouge_scores = []

        for r in results:
            bleu = compute_bleu(r["human_prompt"], r["predicted_prompt"])
            rouge = compute_rouge_l(r["human_prompt"], r["predicted_prompt"])
            r["bleu"] = bleu
            r["rouge_l"] = rouge
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)

        avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
        avg_rouge = sum(rouge_scores) / max(len(rouge_scores), 1)

        print(f"  Avg BLEU-4:    {avg_bleu:.4f}")
        print(f"  Avg ROUGE-L:   {avg_rouge:.4f}")

        # Show examples
        print(f"\n--- Sample Predictions ---")
        for i, r in enumerate(results[:5]):
            gpt_preview = r["gpt_response"][:150] + ("..." if len(r["gpt_response"]) > 150 else "")
            print(f"\n  [{i+1}] GPT response: {gpt_preview}")
            print(f"      Actual:    {r['human_prompt'][:200]}")
            print(f"      Predicted: {r['predicted_prompt'][:200]}")
            print(f"      BLEU={r['bleu']:.3f}  ROUGE-L={r['rouge_l']:.3f}")

        loss_metrics["avg_bleu"] = avg_bleu
        loss_metrics["avg_rouge_l"] = avg_rouge

    # 3. Save results
    output_path = args.output
    if output_path is None:
        output_path = str(Path(config.output_dir) / "eval_results.json")

    output_data = {"metrics": loss_metrics}
    if args.num_generate > 0:
        output_data["predictions"] = results

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
