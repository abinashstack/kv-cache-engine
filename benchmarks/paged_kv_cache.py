"""
Milestone 2: Paged KV cache benchmark.

Generation loop that, at every step, extracts GPT-2's real per-layer KV
tensors, stores them in fixed-size blocks (kv_cache.PagedKVCache), and
reconstructs a fresh past_key_values from those blocks to feed back into
the model. This is a genuine round-trip through paged storage, not a
mock — correctness is verified by asserting the generated token IDs match
the baseline (contiguous HF cache) exactly, since both use greedy decoding.
"""

import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from kv_cache import BlockAllocatorConfig, PagedKVCache

MODEL_NAME = "gpt2"
PROMPT = "The history of artificial intelligence began"
OUTPUT_LENGTHS = [10, 20, 40, 80, 160]
BLOCK_SIZE = 16


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def get_block_config(model) -> BlockAllocatorConfig:
    return BlockAllocatorConfig(
        block_size=BLOCK_SIZE,
        num_blocks=512,
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head,
        head_dim=model.config.n_embd // model.config.n_head,
    )


def _to_storage_format(layer_keys: torch.Tensor, layer_values: torch.Tensor):
    """[1, num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]."""
    return layer_keys.squeeze(0).transpose(0, 1), layer_values.squeeze(0).transpose(0, 1)


def _to_model_format(keys: torch.Tensor, values: torch.Tensor):
    """[seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]."""
    return keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0)


def _rebuild_cache(kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(kv_per_layer):
        model_keys, model_values = _to_model_format(keys, values)
        cache.update(model_keys, model_values, layer_idx)
    return cache


@torch.no_grad()
def generate_with_paged_kv(
    model, input_ids: torch.Tensor, max_new_tokens: int, paged_cache: PagedKVCache
):
    start = time.perf_counter()
    seq_id = paged_cache.new_sequence()
    eos_token_id = model.config.eos_token_id
    peak_used_blocks = 0

    try:
        # Prefill: real forward pass over the prompt, extract and store real KV.
        out = model(input_ids, use_cache=True)
        pkv = out.past_key_values
        keys_per_layer = []
        values_per_layer = []
        for layer in pkv.layers:
            k, v = _to_storage_format(layer.keys, layer.values)
            keys_per_layer.append(k)
            values_per_layer.append(v)
        paged_cache.add_kv(seq_id, keys_per_layer, values_per_layer)
        peak_used_blocks = paged_cache.memory_usage()["used_blocks"]

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        prev_len = keys_per_layer[0].shape[0]

        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            rebuilt = _rebuild_cache(paged_cache.get_kv(seq_id))
            out = model(
                next_token,
                past_key_values=rebuilt,
                use_cache=True,
                cache_position=torch.tensor([prev_len]),
            )

            # Extract only the newly-appended token's KV (last position) and store it.
            new_keys_per_layer = []
            new_values_per_layer = []
            for layer in out.past_key_values.layers:
                k, v = _to_storage_format(
                    layer.keys[:, :, prev_len:, :], layer.values[:, :, prev_len:, :]
                )
                new_keys_per_layer.append(k)
                new_values_per_layer.append(v)
            paged_cache.add_kv(seq_id, new_keys_per_layer, new_values_per_layer)
            peak_used_blocks = max(
                peak_used_blocks, paged_cache.memory_usage()["used_blocks"]
            )
            prev_len += 1

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    finally:
        paged_cache.free_sequence(seq_id)

    return input_ids, time.perf_counter() - start, peak_used_blocks


@torch.no_grad()
def generate_baseline(
    model, input_ids: torch.Tensor, max_new_tokens: int
) -> Tuple[torch.Tensor, float]:
    start = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
    )
    return output_ids, time.perf_counter() - start


def run():
    model, tokenizer = load_model()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    block_config = get_block_config(model)

    results = {"baseline_latency": [], "paged_latency": [], "peak_blocks": []}

    print(f"{'tokens':>8} {'baseline (s)':>14} {'paged (s)':>12} {'overhead':>10}  match")
    for n in OUTPUT_LENGTHS:
        paged_cache = PagedKVCache(block_config)

        baseline_ids, t_baseline = generate_baseline(model, input_ids.clone(), n)
        paged_ids, t_paged, peak_blocks = generate_with_paged_kv(
            model, input_ids.clone(), n, paged_cache
        )

        match = torch.equal(baseline_ids, paged_ids)
        if not match:
            raise AssertionError(
                f"Paged-cache output diverged from baseline at n={n}!\n"
                f"baseline: {tokenizer.decode(baseline_ids[0])}\n"
                f"paged:    {tokenizer.decode(paged_ids[0])}"
            )

        results["baseline_latency"].append(t_baseline)
        results["paged_latency"].append(t_paged)
        results["peak_blocks"].append(peak_blocks)

        overhead = (t_paged / t_baseline - 1) * 100
        print(f"{n:>8} {t_baseline:>14.4f} {t_paged:>12.4f} {overhead:>9.1f}%  {match}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(OUTPUT_LENGTHS, results["baseline_latency"], marker="o", label="Baseline (contiguous cache)")
    axes[0].plot(OUTPUT_LENGTHS, results["paged_latency"], marker="s", label="Paged KV cache")
    axes[0].set_xlabel("output tokens generated")
    axes[0].set_ylabel("wall-clock time (s)")
    axes[0].set_title("Generation Latency: baseline vs. paged")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(OUTPUT_LENGTHS)), results["peak_blocks"], alpha=0.7, color="steelblue")
    axes[1].set_xlabel("output tokens generated")
    axes[1].set_ylabel("blocks used (freed after each run)")
    axes[1].set_title(f"Peak Block Usage (block_size={BLOCK_SIZE} tokens)")
    axes[1].set_xticks(range(len(OUTPUT_LENGTHS)))
    axes[1].set_xticklabels(OUTPUT_LENGTHS)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmarks/paged_kv_result.png", dpi=150, bbox_inches="tight")
    print("\nAll outputs matched baseline exactly (greedy decoding, verified token-for-token).")
    print("Plot saved to benchmarks/paged_kv_result.png")


if __name__ == "__main__":
    run()
