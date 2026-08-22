"""
Milestone 3: continuous batching benchmark.

Runs the same queue of variable-length requests through the scheduler
under two admission policies:

  - static: fixed-size batches; a new batch only forms once the
    previous one has fully drained, so a slot freed by an early-
    finishing request sits idle until every request in its batch is done.
  - continuous: a freed slot is immediately backfilled from the waiting
    queue, so slots are never idle while work remains.

Both policies use the scheduler's incremental KV caching (see
scheduler.py): each round advances every active request by exactly one
token against its own cached history, rather than recomputing full
attention over all past positions every round.

Correctness is verified end-to-end: every request's greedily-generated
text is compared token-for-token against standalone (unbatched,
unpadded) generation with the same prompt and length.
"""

from typing import List

import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from scheduler import Request, Scheduler

MODEL_NAME = "gpt2"
MAX_BATCH_SIZE = 4

# Deliberately skewed lengths: short requests interleaved with long ones,
# to make static batching's idle-slot waste visible.
PROMPTS_AND_LENGTHS = [
    ("The history of artificial intelligence began", 10),
    ("In a distant galaxy, explorers discovered", 80),
    ("The recipe calls for two cups of", 12),
    ("Scientists have long debated whether", 90),
    ("Once upon a time, in a small village", 8),
    ("The stock market reacted sharply after", 70),
    ("According to the latest research,", 15),
    ("The mountain climbers reached the summit and", 85),
]


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def build_requests(tokenizer) -> List[Request]:
    requests = []
    for i, (prompt, max_new) in enumerate(PROMPTS_AND_LENGTHS):
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        requests.append(Request(request_id=i, prompt_ids=prompt_ids, max_new_tokens=max_new))
    return requests


@torch.no_grad()
def verify_against_standalone(model, tokenizer, requests: List[Request]) -> None:
    """Assert every scheduled request's output matches unbatched greedy generation."""
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
        # Both stop early on EOS; compare only up to the shorter of the two.
        n = min(len(standalone_new), len(r.generated_ids))
        if standalone_new[:n] != r.generated_ids[:n]:
            raise AssertionError(
                f"Request {r.request_id} diverged from standalone generation!\n"
                f"standalone: {tokenizer.decode(standalone_new)}\n"
                f"scheduled:  {tokenizer.decode(r.generated_ids)}"
            )


def run_policy(model, tokenizer, policy: str):
    requests = build_requests(tokenizer)
    scheduler = Scheduler(
        model,
        eos_token_id=model.config.eos_token_id,
        max_batch_size=MAX_BATCH_SIZE,
        policy=policy,
    )
    for r in requests:
        scheduler.add_request(r)

    finished = scheduler.run()
    latencies = [r.finish_step for r in finished]  # already wall-clock seconds
    finished_sorted = sorted(finished, key=lambda r: r.request_id)

    return finished_sorted, scheduler.wall_time, latencies, scheduler.num_decode_rounds


def run():
    model, tokenizer = load_model()

    print("Verifying correctness against standalone generation...")
    ref_requests = build_requests(tokenizer)
    ref_scheduler = Scheduler(
        model,
        eos_token_id=model.config.eos_token_id,
        max_batch_size=MAX_BATCH_SIZE,
        policy="continuous",
    )
    for r in ref_requests:
        ref_scheduler.add_request(r)
    ref_finished = ref_scheduler.run()
    verify_against_standalone(model, tokenizer, ref_finished)
    print("All requests match standalone greedy generation exactly.\n")

    results = {}
    print(f"{'policy':>12} {'makespan (s)':>14} {'rounds':>8} {'avg latency (s)':>18}")
    for policy in ("static", "continuous"):
        finished, makespan, latencies, num_rounds = run_policy(model, tokenizer, policy)
        avg_latency = sum(latencies) / len(latencies)
        results[policy] = {
            "makespan": makespan,
            "latencies": latencies,
            "num_rounds": num_rounds,
        }
        print(f"{policy:>12} {makespan:>14.3f} {num_rounds:>8} {avg_latency:>18.3f}")

    speedup = results["static"]["makespan"] / results["continuous"]["makespan"]
    print(f"\nContinuous batching makespan speedup: {speedup:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    policies = ["static", "continuous"]
    makespans = [results[p]["makespan"] for p in policies]
    axes[0].bar(policies, makespans, alpha=0.7, color=["indianred", "steelblue"])
    axes[0].set_ylabel("wall-clock time (s)")
    axes[0].set_title("Makespan: time to drain full request queue")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].boxplot(
        [results["static"]["latencies"], results["continuous"]["latencies"]],
        tick_labels=policies,
    )
    axes[1].set_ylabel("per-request completion latency (s)")
    axes[1].set_title("Per-request latency distribution")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmarks/continuous_batching_result.png", dpi=150, bbox_inches="tight")
    print("Plot saved to benchmarks/continuous_batching_result.png")


if __name__ == "__main__":
    run()
