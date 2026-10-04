"""Real-model check: batched/prefix-cached generation must match plain greedy
generation token for token. Skipped when GPT-2 can't be loaded (e.g. offline)."""

import pytest
import torch

from prefix_cache import PrefixCache
from scheduler import Request, Scheduler


@pytest.fixture(scope="module")
def lm():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    except Exception as exc:
        pytest.skip(f"gpt2 unavailable: {exc}")
    model.eval()
    return model, tok


def _greedy(model, ids, n):
    t = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=n,
                             do_sample=False, pad_token_id=model.config.eos_token_id)
    return out[0, len(ids):].tolist()


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_batched_matches_standalone(lm, policy):
    model, tok = lm
    prompts = ["The history of", "Once upon a time, in a small village", "Hello"]
    reqs = [Request(i, tok(p).input_ids, 8) for i, p in enumerate(prompts)]
    sched = Scheduler(model, model.config.eos_token_id, max_batch_size=2, policy=policy)
    for r in reqs:
        sched.add_request(r)
    sched.run()
    for r in reqs:
        expect = _greedy(model, r.prompt_ids, 8)
        assert r.generated_ids == expect[: len(r.generated_ids)]


def test_prefix_cache_hit_matches_miss(lm):
    model, tok = lm
    prefix = tok("You are a helpful assistant.").input_ids
    suffixes = [tok(" What is 2+2?").input_ids, tok(" Name a color.").input_ids]
    cache = PrefixCache()
    reqs = [Request(i, prefix + s, 6, shared_prefix_len=len(prefix)) for i, s in enumerate(suffixes)]
    sched = Scheduler(model, model.config.eos_token_id, 1, "continuous", prefix_cache=cache)
    for r in reqs:
        sched.add_request(r)
    sched.run()
    assert (cache.hits, cache.misses) == (1, 1)
    for r in reqs:
        assert r.generated_ids == _greedy(model, r.prompt_ids, 6)[: len(r.generated_ids)]
