"""
Milestone 1: baseline benchmark.

Measures generation latency for GPT-2 small with and without KV caching,
across increasing output lengths. This is the reference point every later
milestone (paged cache, continuous batching, prefix caching) gets compared
against.
"""

import time

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"
PROMPT = "The history of artificial intelligence began"
OUTPUT_LENGTHS = [10, 20, 40, 80, 160]


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def time_generation(model, input_ids, max_new_tokens, use_cache):
    start = time.perf_counter()
    model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
    )
    return time.perf_counter() - start


def run():
    model, tokenizer = load_model()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    results = {"cached": [], "uncached": []}
    print(f"{'tokens':>8} {'cached (s)':>12} {'uncached (s)':>14} {'speedup':>10}")
    for n in OUTPUT_LENGTHS:
        t_cached = time_generation(model, input_ids, n, use_cache=True)
        t_uncached = time_generation(model, input_ids, n, use_cache=False)
        results["cached"].append(t_cached)
        results["uncached"].append(t_uncached)
        print(f"{n:>8} {t_cached:>12.3f} {t_uncached:>14.3f} {t_uncached / t_cached:>9.2f}x")

    plt.plot(OUTPUT_LENGTHS, results["cached"], marker="o", label="KV cache (use_cache=True)")
    plt.plot(OUTPUT_LENGTHS, results["uncached"], marker="o", label="no cache (use_cache=False)")
    plt.xlabel("output tokens generated")
    plt.ylabel("wall-clock time (s)")
    plt.title("GPT-2 small generation latency: cached vs uncached")
    plt.legend()
    plt.savefig("benchmarks/baseline_result.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to benchmarks/baseline_result.png")


if __name__ == "__main__":
    run()
