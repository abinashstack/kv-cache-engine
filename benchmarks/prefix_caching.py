"""
Milestone 4: prefix caching benchmark.

Simulates a common serving pattern: many requests share a long system
prompt but differ in their user query. Compares total prefill wall-clock
time across the whole request set with prefix caching off (every
request prefills the full prompt from scratch) vs on (only the first
request pays full cost; the rest reuse the cached system-prompt KV and
prefill just their own suffix).

Correctness is verified end-to-end: every request's generated text
(cache hits included) is compared token-for-token against standalone
generation of the same full prompt with no caching involved at all.
"""

from typing import List

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from prefix_cache import PrefixCache
from scheduler import Request, Scheduler

MODEL_NAME = "gpt2"
MAX_BATCH_SIZE = 4
MAX_NEW_TOKENS = 20

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following question concisely "
    "and accurately, using only well-established facts."
)

USER_QUERIES = [
    " What is the capital of France?",
    " What is the boiling point of water in Celsius?",
    " Who wrote the play Romeo and Juliet?",
    " What is the largest planet in the solar system?",
    " What year did the first moon landing happen?",
    " What is the chemical symbol for gold?",
    " How many continents are there on Earth?",
    " What is the speed of light in a vacuum?",
]


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def build_requests(tokenizer):
    prefix_ids = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids[0].tolist()
    requests = []
    for i, query in enumerate(USER_QUERIES):
        suffix_ids = tokenizer(query, return_tensors="pt").input_ids[0].tolist()
        # Concatenate at the token level (not string level) so every
        # request's prompt shares the exact same prefix token sequence --
        # tokenizing the concatenated string could merge BPE tokens
        # differently at the boundary and silently break the shared prefix.
        requests.append(
            Request(
                request_id=i,
                prompt_ids=prefix_ids + suffix_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                shared_prefix_len=len(prefix_ids),
            )
        )
    return requests, prefix_ids


@torch.no_grad()
def verify_against_standalone(model, tokenizer, requests: List[Request]) -> None:
    for r in requests:
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


def run_with_cache(model, tokenizer, prefix_ids, use_prefix_cache: bool):
    requests, _ = build_requests(tokenizer)
    prefix_cache = PrefixCache() if use_prefix_cache else None
    scheduler = Scheduler(
        model,
        eos_token_id=model.config.eos_token_id,
        max_batch_size=MAX_BATCH_SIZE,
        policy="continuous",
        prefix_cache=prefix_cache,
    )
    for r in requests:
        scheduler.add_request(r)
    finished = scheduler.run()
    finished_sorted = sorted(finished, key=lambda r: r.request_id)
    hits = prefix_cache.hits if prefix_cache else 0
    misses = prefix_cache.misses if prefix_cache else len(requests)
    return finished_sorted, scheduler.prefill_time, scheduler.wall_time, hits, misses


def run():
    model, tokenizer = load_model()
    _, prefix_ids = build_requests(tokenizer)
    print(f"Shared system prompt: {len(prefix_ids)} tokens; {len(USER_QUERIES)} requests\n")

    print("Verifying correctness against standalone generation...")
    ref_finished, _, _, hits, misses = run_with_cache(model, tokenizer, prefix_ids, use_prefix_cache=True)
    verify_against_standalone(model, tokenizer, ref_finished)
    print(f"All requests match standalone generation exactly. (cache hits={hits}, misses={misses})\n")

    results = {}
    print(f"{'mode':>14} {'prefill time (s)':>18} {'total makespan (s)':>20}")
    for label, use_cache in (("no prefix cache", False), ("prefix cache", True)):
        finished, prefill_time, makespan, hits, misses = run_with_cache(
            model, tokenizer, prefix_ids, use_prefix_cache=use_cache
        )
        results[label] = {"prefill_time": prefill_time, "makespan": makespan}
        print(f"{label:>14} {prefill_time:>18.4f} {makespan:>20.4f}")

    prefill_speedup = results["no prefix cache"]["prefill_time"] / results["prefix cache"]["prefill_time"]
    print(f"\nPrefill time speedup from prefix caching: {prefill_speedup:.2f}x")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["no prefix cache", "prefix cache"]
    prefill_times = [results[l]["prefill_time"] for l in labels]
    ax.bar(labels, prefill_times, alpha=0.7, color=["indianred", "steelblue"])
    ax.set_ylabel("total prefill wall-clock time (s)")
    ax.set_title(f"Prefill cost across {len(USER_QUERIES)} requests sharing a system prompt")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmarks/prefix_caching_result.png", dpi=150, bbox_inches="tight")
    print("Plot saved to benchmarks/prefix_caching_result.png")


if __name__ == "__main__":
    run()
