"""
Tests to validate that seed-based sampling produces diverse outputs across GPUs.

Loads Qwen on one GPU and generates 5 outputs per prompt (simulating 5 GPUs with
different seeds), then checks diversity metrics.

Usage:
  pytest test_t2t_sampling_diversity.py -v
"""

import difflib

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ketan/LLMs/models/Qwen_Qwen2.5-7B-Instruct/"
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
BASE_SEED = 42
NUM_SEEDS = 5
MAX_NEW_TOKENS = 256

TEST_PROMPTS = [
    "Explain the concept of recursion in programming.",
    "What are the main differences between Python and C++?",
    "Write a short poem about the ocean.",
    "How does a neural network learn?",
    "What is the significance of the Turing test?",
    "Describe the process of photosynthesis.",
    "What are the pros and cons of remote work?",
    "Explain quantum entanglement in simple terms.",
    "How do vaccines work?",
    "What is the trolley problem in philosophy?",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all tests."""
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@pytest.fixture(scope="module")
def generated_outputs(model_and_tokenizer):
    """Generate NUM_SEEDS outputs per prompt using different seeds."""
    model, tokenizer = model_and_tokenizer
    results = {}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.05,
    )

    for prompt in TEST_PROMPTS:
        outputs = []
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024,
        ).to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        for offset in range(NUM_SEEDS):
            torch.manual_seed(BASE_SEED + offset)
            torch.cuda.manual_seed(BASE_SEED + offset)

            with torch.no_grad():
                gen = model.generate(**inputs, **gen_kwargs)

            decoded = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            outputs.append(decoded)

        results[prompt] = outputs

    return results


class TestBasicDiversity:
    """Test that outputs from different seeds are not identical."""

    def test_not_all_identical(self, generated_outputs):
        """At least one prompt should have non-identical outputs across seeds."""
        for prompt, outputs in generated_outputs.items():
            unique = set(outputs)
            if len(unique) > 1:
                return
        pytest.fail("All prompts produced identical outputs across all seeds")

    def test_unique_ratio_above_threshold(self, generated_outputs):
        """Each prompt should have >= 60% unique outputs across seeds."""
        for prompt, outputs in generated_outputs.items():
            unique_ratio = len(set(outputs)) / len(outputs)
            assert unique_ratio >= 0.6, (
                f"Prompt '{prompt[:50]}...' has only {unique_ratio:.0%} unique outputs "
                f"({len(set(outputs))}/{len(outputs)})"
            )


class TestStatisticalDiversity:
    """Test diversity using pairwise similarity metrics."""

    def test_average_pairwise_char_difference(self, generated_outputs):
        """Average pairwise SequenceMatcher similarity should be < 0.95."""
        for prompt, outputs in generated_outputs.items():
            similarities = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    ratio = difflib.SequenceMatcher(
                        None, outputs[i], outputs[j]
                    ).ratio()
                    similarities.append(ratio)
            avg_sim = sum(similarities) / len(similarities)
            assert avg_sim < 0.95, (
                f"Prompt '{prompt[:50]}...' has avg pairwise similarity {avg_sim:.3f} >= 0.95"
            )

    def test_aggregate_diversity(self, generated_outputs):
        """Across all prompts, average unique ratio should be >= 0.7."""
        ratios = []
        for prompt, outputs in generated_outputs.items():
            ratios.append(len(set(outputs)) / len(outputs))
        avg_ratio = sum(ratios) / len(ratios)
        assert avg_ratio >= 0.7, (
            f"Aggregate unique ratio {avg_ratio:.3f} < 0.7"
        )


class TestOutputQuality:
    """Test that outputs are valid and non-degenerate."""

    def test_outputs_non_empty(self, generated_outputs):
        """All outputs should be non-empty."""
        for prompt, outputs in generated_outputs.items():
            for i, output in enumerate(outputs):
                assert len(output.strip()) > 0, (
                    f"Empty output for prompt '{prompt[:50]}...' at seed offset {i}"
                )

    def test_outputs_not_too_short(self, generated_outputs):
        """Less than 20% of outputs should be shorter than 10 chars."""
        total = 0
        too_short = 0
        for prompt, outputs in generated_outputs.items():
            for output in outputs:
                total += 1
                if len(output.strip()) < 10:
                    too_short += 1
        ratio = too_short / total
        assert ratio < 0.2, (
            f"{too_short}/{total} ({ratio:.0%}) outputs are shorter than 10 chars"
        )
