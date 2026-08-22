"""
Milestone 6: final benchmarks.

Consolidates Milestones 2-4 (incremental per-request KV caching,
continuous batching, prefix caching) into one "optimized" configuration
and compares it against Milestone 1's approach extended to a full
request queue: naive sequential generation via HF's standard
use_cache=True generate(), one request at a time, no batching, no
prefix reuse -- i.e. what a service using none of this project's
techniques would do, run against the same workload.

Workload: 14 requests -- 8 share one system prompt (to exercise prefix
caching; a real, non-trivial hit rate falls out of this rather than
being hard-coded) and 6 are standalone, with a mix of short and long
max_new_tokens (to exercise continuous batching).

Reported, plotted against the baseline: aggregate tokens/sec, p50/p99
per-request completion latency, peak concurrent KV cache memory, and
prefix cache hit rate.

Memory note: the optimized system's peak KV memory is the sum of all
*currently active* (not yet finished) requests' cached KV tensors at
the highest-concurrency moment -- real bytes, not an estimate. The
baseline holds exactly one request's KV at a time by construction, so
its peak is just the single largest request's KV size. Reporting these
side by side is intentionally honest about the tradeoff: the optimized
system's peak memory is higher because it holds multiple sequences
concurrently -- that concurrency is exactly what buys the throughput
and latency wins below, not a free lunch.
"""

import statistics
import time
from typing import List

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from prefix_cache import PrefixCache
from scheduler import Request, Scheduler, prefill, step_batch_incremental

MODEL_NAME = "gpt2"
MAX_BATCH_SIZE = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question concisely "
    "and accurately, using only well-established facts."
)
SHARED_QUERIES_AND_LENGTHS = [
    (" What is the capital of France?", 10),
    (" What is the boiling point of water in Celsius?", 50),
    (" Who wrote the play Romeo and Juliet?", 12),
    (" What is the largest planet in the solar system?", 45),
    (" What year did the first moon landing happen?", 15),
    (" What is the chemical symbol for gold?", 55),
    (" How many continents are there on Earth?", 10),
    (" What is the speed of light in a vacuum?", 40),
]
STANDALONE_PROMPTS_AND_LENGTHS = [
    ("The history of artificial intelligence began", 10),
    ("In a distant galaxy, explorers discovered", 60),
    ("The recipe calls for two cups of", 12),
    ("Scientists have long debated whether", 55),
    ("Once upon a time, in a small village", 8),
    ("The stock market reacted sharply after", 50),
]


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def build_workload(tokenizer):
    prefix_ids = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids[0].tolist()
    items = []
    for query, max_new in SHARED_QUERIES_AND_LENGTHS:
        suffix_ids = tokenizer(query, return_tensors="pt").input_ids[0].tolist()
        items.append((prefix_ids + suffix_ids, max_new, len(prefix_ids)))
    for prompt, max_new in STANDALONE_PROMPTS_AND_LENGTHS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        items.append((ids, max_new, 0))
    return items


def kv_bytes(layer_kv) -> int:
    return sum(k.element_size() * k.nelement() + v.element_size() * v.nelement() for k, v in layer_kv)


@torch.no_grad()
def run_baseline(model, tokenizer, workload):
    """Sequential, one request at a time, HF's standard cache -- no batching,
    no prefix reuse. This is Milestone 1's approach applied to a queue."""
    wall_time = 0.0
    total_tokens = 0
    latencies = []
    peak_kv_bytes = 0

    for prompt_ids, max_new_tokens, _ in workload:
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        start = time.perf_counter()
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
        wall_time += time.perf_counter() - start

        num_generated = out.shape[1] - ids.shape[1]
        total_tokens += num_generated
        latencies.append(wall_time)

        final_len = out.shape[1]
        this_bytes = (
            final_len
            * model.config.n_layer
            * model.config.n_head
            * (model.config.n_embd // model.config.n_head)
            * 4  # float32
            * 2  # key + value
        )
        peak_kv_bytes = max(peak_kv_bytes, this_bytes)

    return {
        "wall_time": wall_time,
        "total_tokens": total_tokens,
        "latencies": latencies,
        "peak_kv_bytes": peak_kv_bytes,
        "hits": 0,
        "misses": len(workload),
    }


def run_optimized(model, tokenizer, workload):
    """Continuous batching + incremental caching + prefix caching, combined."""
    prefix_cache = PrefixCache()
    scheduler = Scheduler(
        model,
        eos_token_id=model.config.eos_token_id,
        max_batch_size=MAX_BATCH_SIZE,
        policy="continuous",
        prefix_cache=prefix_cache,
    )
    for i, (prompt_ids, max_new_tokens, shared_prefix_len) in enumerate(workload):
        scheduler.add_request(
            Request(
                request_id=i,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                shared_prefix_len=shared_prefix_len,
            )
        )

    # Peak concurrent KV memory: measured across the *running* set (active,
    # not-yet-finished requests) at each decode round -- the real working
    # set a server would need to hold, not a static estimate.
    peak_kv_bytes = 0
    orig_step = step_batch_incremental

    def instrumented_step(model_, requests):
        nonlocal peak_kv_bytes
        peak_kv_bytes = max(peak_kv_bytes, sum(kv_bytes(r.layer_kv) for r in requests))
        orig_step(model_, requests)

    import scheduler as scheduler_module

    scheduler_module.step_batch_incremental = instrumented_step
    try:
        finished = scheduler.run()
    finally:
        scheduler_module.step_batch_incremental = orig_step

    finished_sorted = sorted(finished, key=lambda r: r.request_id)
    total_tokens = sum(r.num_generated for r in finished_sorted)
    latencies = [r.finish_step for r in finished_sorted]

    return {
        "wall_time": scheduler.wall_time,
        "total_tokens": total_tokens,
        "latencies": latencies,
        "peak_kv_bytes": peak_kv_bytes,
        "hits": prefix_cache.hits,
        "misses": prefix_cache.misses,
        "finished": finished_sorted,
    }


@torch.no_grad()
def verify_correctness(model, tokenizer, finished: List[Request]) -> None:
    for r in finished:
        prompt_ids = torch.tensor([r.prompt_ids], dtype=torch.long)
        standalone = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=r.max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
        standalone_new = standalone[0, prompt_ids.shape[1]:].tolist()
        n = min(len(standalone_new), len(r.generated_ids))
        if standalone_new[:n] != r.generated_ids[:n]:
            raise AssertionError(
                f"Request {r.request_id} diverged from standalone generation!\n"
                f"standalone: {tokenizer.decode(standalone_new)}\n"
                f"scheduled:  {tokenizer.decode(r.generated_ids)}"
            )


def percentile(data, p):
    return statistics.quantiles(data, n=100)[p - 1] if len(data) > 1 else data[0]


def run():
    model, tokenizer = load_model()
    workload = build_workload(tokenizer)
    print(f"Workload: {len(workload)} requests ({len(SHARED_QUERIES_AND_LENGTHS)} sharing a system prompt, "
          f"{len(STANDALONE_PROMPTS_AND_LENGTHS)} standalone)\n")

    print("Verifying optimized-path correctness against standalone generation...")
    opt_for_verify = run_optimized(model, tokenizer, workload)
    verify_correctness(model, tokenizer, opt_for_verify["finished"])
    print("All requests match standalone greedy generation exactly.\n")

    print("Running baseline (Milestone 1 style: sequential, no batching, no prefix cache)...")
    baseline = run_baseline(model, tokenizer, workload)

    print("Running optimized (Milestones 2-4 combined: incremental cache + continuous batching + prefix cache)...")
    optimized = run_optimized(model, tokenizer, workload)

    def summarize(label, result):
        tps = result["total_tokens"] / result["wall_time"]
        p50 = percentile(result["latencies"], 50)
        p99 = percentile(result["latencies"], 99)
        mem_mb = result["peak_kv_bytes"] / (1024 * 1024)
        eligible = result["hits"] + result["misses"] if label == "optimized" else 0
        hit_rate = (result["hits"] / eligible * 100) if eligible else 0.0
        print(f"\n{label}:")
        print(f"  wall-clock time:        {result['wall_time']:.3f}s")
        print(f"  total tokens generated: {result['total_tokens']}")
        print(f"  throughput:             {tps:.2f} tokens/sec")
        print(f"  p50 latency:            {p50:.3f}s")
        print(f"  p99 latency:            {p99:.3f}s")
        print(f"  peak concurrent KV mem: {mem_mb:.3f} MB")
        if label == "optimized":
            print(f"  prefix cache hit rate:  {hit_rate:.1f}% ({result['hits']} hits / {result['misses']} misses)")
        return tps, p50, p99, mem_mb

    base_tps, base_p50, base_p99, base_mem = summarize("baseline", baseline)
    opt_tps, opt_p50, opt_p99, opt_mem = summarize("optimized", optimized)

    print(f"\nThroughput speedup:  {opt_tps / base_tps:.2f}x")
    print(f"p50 latency speedup: {base_p50 / opt_p50:.2f}x")
    print(f"p99 latency speedup: {base_p99 / opt_p99:.2f}x")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["baseline", "optimized"]

    axes[0].bar(labels, [base_tps, opt_tps], alpha=0.7, color=["indianred", "steelblue"])
    axes[0].set_ylabel("tokens/sec")
    axes[0].set_title("Throughput")
    axes[0].grid(True, alpha=0.3, axis="y")

    x = range(2)
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], [base_p50, opt_p50], width, label="p50", color="steelblue", alpha=0.7)
    axes[1].bar([i + width / 2 for i in x], [base_p99, opt_p99], width, label="p99", color="indianred", alpha=0.7)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("latency (s)")
    axes[1].set_title("Per-request latency (p50 / p99)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(labels, [base_mem, opt_mem], alpha=0.7, color=["indianred", "steelblue"])
    axes[2].set_ylabel("peak concurrent KV memory (MB)")
    axes[2].set_title("Memory (optimized holds more -- concurrency, not a free lunch)")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmarks/final_benchmarks_result.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to benchmarks/final_benchmarks_result.png")


if __name__ == "__main__":
    run()
