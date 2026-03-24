"""
Interactive testing pipeline for the inverse mapping model.

Round-trip: GPT response → predicted human prompt → LLaMA response.

Usage:
    python test_pipeline.py
    python test_pipeline.py --inverse-device cuda:0 --base-device cuda:1
    python test_pipeline.py --prompt "Here is a GPT response to process..."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defaults from training config
INVERSE_MODEL_PATH = "checkpoints/inverse_mapping/final_model"
BASE_MODEL_PATH = "/home/ketan/LLMs/models/meta-llama_Llama-3.1-8B"
SEPARATOR_TOKEN_ID = 128002  # <|reserved_special_token_0|>
PAD_TOKEN_ID = 128004        # <|finetune_right_pad_id|>


def load_models(inverse_path, base_path, inverse_device, base_device):
    """Load both models and the shared tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token_id = PAD_TOKEN_ID

    print(f"Loading inverse mapping model → {inverse_device}")
    inverse_model = AutoModelForCausalLM.from_pretrained(
        inverse_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(inverse_device).eval()

    print(f"Loading base LLaMA model → {base_device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(base_device).eval()

    print("Both models loaded.\n")
    return inverse_model, base_model, tokenizer


@torch.no_grad()
def predict_prompt(model, tokenizer, gpt_response, device,
                   max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Inverse model: GPT response → predicted human prompt.

    Input format matches training: [BOS] gpt_response [SEP] → generates prompt tokens until [EOS].
    """
    gpt_ids = tokenizer.encode(gpt_response, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + gpt_ids + [SEPARATOR_TOKEN_ID]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the generated part (after input)
    generated = output_ids[0, len(input_ids):]
    if len(generated) > 0 and generated[-1] == tokenizer.eos_token_id:
        generated = generated[:-1]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_response(model, tokenizer, prompt, device,
                      max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Base LLaMA: human prompt → model completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_pipeline(inverse_model, base_model, tokenizer, gpt_response,
                 inverse_device, base_device, **gen_kwargs):
    """Full round-trip: GPT response → predicted prompt → LLaMA response."""
    predicted_prompt = predict_prompt(
        inverse_model, tokenizer, gpt_response, inverse_device, **gen_kwargs,
    )
    llama_response = generate_response(
        base_model, tokenizer, predicted_prompt, base_device, **gen_kwargs,
    )
    return predicted_prompt, llama_response


def print_results(gpt_response, predicted_prompt, llama_response):
    """Display the three-way comparison."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  1. INPUT (GPT Response)")
    print(sep)
    print(gpt_response)
    print(f"\n{sep}")
    print("  2. PREDICTED HUMAN PROMPT (Inverse Model)")
    print(sep)
    print(predicted_prompt)
    print(f"\n{sep}")
    print("  3. LLaMA RESPONSE (Base Model → Predicted Prompt)")
    print(sep)
    print(llama_response)
    print(sep)


def interactive_loop(inverse_model, base_model, tokenizer,
                     inverse_device, base_device, gen_kwargs):
    """Read GPT responses from stdin and run the pipeline."""
    print("Enter a GPT/model response. The inverse model will predict the")
    print("human prompt that produced it, then LLaMA will respond to that prompt.")
    print("Type 'quit' to exit. Enter '---' for multi-line input.\n")

    while True:
        try:
            gpt_response = read_input()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if gpt_response is None:
            break
        if not gpt_response.strip():
            continue

        print("\nRunning inverse model...")
        predicted_prompt = predict_prompt(
            inverse_model, tokenizer, gpt_response, inverse_device,
            **gen_kwargs,
        )
        print("Running base LLaMA...")
        llama_response = generate_response(
            base_model, tokenizer, predicted_prompt, base_device,
            **gen_kwargs,
        )
        print_results(gpt_response, predicted_prompt, llama_response)
        print()


def read_input():
    """Read single- or multi-line input from the user."""
    print("GPT Response> ", end="", flush=True)
    first_line = input()

    if first_line.strip().lower() in ("quit", "exit", "q"):
        return None

    if first_line.strip() == "---":
        lines = []
        print("  (multi-line mode — enter '---' on its own line to finish)")
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == "---":
                break
            lines.append(line)
        return "\n".join(lines)

    return first_line


def main():
    parser = argparse.ArgumentParser(
        description="Round-trip test: GPT response → predicted prompt → LLaMA response",
    )
    parser.add_argument("--inverse-model", default=INVERSE_MODEL_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    parser.add_argument("--inverse-device", default="cuda:0")
    parser.add_argument("--base-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single GPT response to process (skip interactive mode)")
    args = parser.parse_args()

    inverse_model, base_model, tokenizer = load_models(
        args.inverse_model, args.base_model,
        args.inverse_device, args.base_device,
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.prompt:
        predicted, response = run_pipeline(
            inverse_model, base_model, tokenizer, args.prompt,
            args.inverse_device, args.base_device, **gen_kwargs,
        )
        print_results(args.prompt, predicted, response)
    else:
        interactive_loop(
            inverse_model, base_model, tokenizer,
            args.inverse_device, args.base_device, gen_kwargs,
        )


if __name__ == "__main__":
    main()
