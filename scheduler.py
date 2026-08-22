"""
Milestone 3: continuous batching scheduler with incremental KV caching.

Iteration-level scheduling: at every generated token, the scheduler
re-evaluates which requests are active, admits newly-arrived requests
into any free slot, and evicts finished ones -- instead of waiting for
an entire fixed-size batch to complete (static batching) before
admitting new work.

Unlike a naive recompute-every-round implementation, each active
request keeps its own real per-layer KV history (built once via a
single-sequence prefill on admission, exactly as in Milestone 2). Every
decode round, the currently active requests' histories are assembled
into one left-padded batch, fed through a single fused forward call to
produce exactly one new token per request, and the newly-appended
per-token KV is split back out and appended to each request's own
history. This is what actually lets continuous batching win on wall-
clock time: per round it does O(batch) new-token attention against
cached history instead of recomputing every past position's hidden
states from scratch.

Batch composition math (left-padding, attention_mask, position_ids for
the new token) was validated against independent standalone generation
before this was written: two prompts of different lengths advanced
together for 6 rounds produced token-for-token identical output to
generating each alone.

Milestone 4 (prefix caching, see prefix_cache.py) plugs into admission
here via prefill_with_prefix_cache: when requests share a token prefix
(e.g. a system prompt), only the first pays full prefill cost; the rest
reuse its cached KV and prefill just their own suffix.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from prefix_cache import PrefixCache

LayerKV = Tuple[torch.Tensor, torch.Tensor]  # (keys, values), each [num_heads, seq_len, head_dim]


@dataclass
class Request:
    request_id: int
    prompt_ids: List[int]
    max_new_tokens: int
    generated_ids: List[int] = field(default_factory=list)
    shared_prefix_len: int = 0  # >0 to probe prefix_cache for this many leading tokens
    layer_kv: Optional[List[LayerKV]] = None  # set on prefill
    start_step: Optional[int] = None
    finish_step: Optional[float] = None  # wall-clock seconds, not a round index
    done: bool = False

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def cur_len(self) -> int:
        return self.layer_kv[0][0].shape[1]


@torch.no_grad()
def prefill(model, request: Request) -> None:
    """Single-sequence forward over the prompt: seeds real per-layer KV and
    produces the request's first generated token."""
    ids = torch.tensor([request.prompt_ids], dtype=torch.long)
    out = model(ids, use_cache=True)
    request.layer_kv = [
        (layer.keys.squeeze(0).clone(), layer.values.squeeze(0).clone())
        for layer in out.past_key_values.layers
    ]
    next_token = torch.argmax(out.logits[0, -1, :]).item()
    request.generated_ids.append(next_token)


@torch.no_grad()
def prefill_with_prefix_cache(
    model, request: Request, prefix_cache: PrefixCache, shared_prefix_len: int
) -> None:
    """Like prefill(), but if request.prompt_ids starts with a prefix already
    in prefix_cache, reuses that prefix's KV and only runs the forward pass
    over the request's unique suffix -- skipping attention/FFN compute for
    the shared tokens entirely. On a cache miss, prefills the whole prompt
    as normal and stores the prefix portion for future requests.

    Validated before use: continuing from a cached prefix produces the same
    first token and the same combined KV (to float32 rounding, ~1e-5) as a
    single-shot prefill over the full prompt.
    """
    prefix_ids = request.prompt_ids[:shared_prefix_len]
    cached_prefix_kv = prefix_cache.get(prefix_ids) if shared_prefix_len else None

    if cached_prefix_kv is None:
        prefill(model, request)
        if shared_prefix_len:
            prefix_slice = [(k[:, :shared_prefix_len, :], v[:, :shared_prefix_len, :]) for k, v in request.layer_kv]
            prefix_cache.put(prefix_ids, prefix_slice)
        return

    suffix_ids = request.prompt_ids[shared_prefix_len:]
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(cached_prefix_kv):
        cache.update(k.unsqueeze(0), v.unsqueeze(0), layer_idx)

    input_ids = torch.tensor([suffix_ids], dtype=torch.long)
    total_len = shared_prefix_len + len(suffix_ids)
    attention_mask = torch.ones(1, total_len, dtype=torch.long)
    position_ids = torch.arange(shared_prefix_len, total_len).unsqueeze(0)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    request.layer_kv = [
        (layer.keys.squeeze(0).clone(), layer.values.squeeze(0).clone())
        for layer in out.past_key_values.layers
    ]
    next_token = torch.argmax(out.logits[0, -1, :]).item()
    request.generated_ids.append(next_token)


@torch.no_grad()
def step_batch_incremental(model, requests: List[Request]) -> None:
    """Advance every request in `requests` by exactly one token, via one
    fused forward call over each request's real cached history (no
    recompute of past positions)."""
    num_layers = len(requests[0].layer_kv)
    num_heads, _, head_dim = requests[0].layer_kv[0][0].shape
    batch = len(requests)
    cur_lens = [r.cur_len for r in requests]
    max_hist = max(cur_lens)

    past_mask = torch.zeros(batch, max_hist, dtype=torch.long)
    for i, cl in enumerate(cur_lens):
        past_mask[i, max_hist - cl:] = 1

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        padded_k = torch.zeros(batch, num_heads, max_hist, head_dim)
        padded_v = torch.zeros(batch, num_heads, max_hist, head_dim)
        for i, r in enumerate(requests):
            k, v = r.layer_kv[layer_idx]
            cl = cur_lens[i]
            padded_k[i, :, max_hist - cl:, :] = k
            padded_v[i, :, max_hist - cl:, :] = v
        cache.update(padded_k, padded_v, layer_idx)

    next_input = torch.tensor([[r.generated_ids[-1]] for r in requests], dtype=torch.long)
    full_mask = torch.cat([past_mask, torch.ones(batch, 1, dtype=torch.long)], dim=1)
    # Position of the new token = count of real (non-padding) past tokens for that row.
    position_ids = past_mask.sum(dim=1, keepdim=True)

    out = model(
        input_ids=next_input,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

    for i, r in enumerate(requests):
        r.generated_ids.append(next_tokens[i].item())
        new_layer_kv = []
        for layer_idx in range(num_layers):
            new_k = out.past_key_values.layers[layer_idx].keys[i, :, -1:, :].clone()
            new_v = out.past_key_values.layers[layer_idx].values[i, :, -1:, :].clone()
            old_k, old_v = r.layer_kv[layer_idx]
            new_layer_kv.append((torch.cat([old_k, new_k], dim=1), torch.cat([old_v, new_v], dim=1)))
        r.layer_kv = new_layer_kv


class Scheduler:
    """Drives a queue of requests to completion under a fixed concurrency
    cap, using either continuous or static batch admission."""

    def __init__(
        self,
        model,
        eos_token_id: int,
        max_batch_size: int,
        policy: str,
        prefix_cache: Optional[PrefixCache] = None,
    ):
        assert policy in ("continuous", "static")
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_batch_size = max_batch_size
        self.policy = policy
        self.prefix_cache = prefix_cache
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []
        self.wall_time = 0.0  # cumulative seconds spent in prefill + decode
        self.prefill_time = 0.0  # subset of wall_time spent in prefill specifically
        self.num_decode_rounds = 0

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def _finish(self, r: Request) -> None:
        r.finish_step = self.wall_time
        r.done = True
        self.finished.append(r)

    def _admit(self) -> None:
        if self.policy == "continuous":
            slots = self.max_batch_size - len(self.running)
            to_admit = self.waiting[:slots]
            del self.waiting[:slots]
        else:  # static: only form a new batch once the previous one is fully drained
            if self.running or not self.waiting:
                to_admit = []
            else:
                n = min(self.max_batch_size, len(self.waiting))
                to_admit = self.waiting[:n]
                del self.waiting[:n]

        for r in to_admit:
            start = time.perf_counter()
            if self.prefix_cache is not None and r.shared_prefix_len > 0:
                prefill_with_prefix_cache(self.model, r, self.prefix_cache, r.shared_prefix_len)
            else:
                prefill(self.model, r)
            elapsed = time.perf_counter() - start
            self.wall_time += elapsed
            self.prefill_time += elapsed

            hit_eos = r.generated_ids[-1] == self.eos_token_id
            hit_max = r.num_generated >= r.max_new_tokens
            if hit_eos or hit_max:
                self._finish(r)
            else:
                self.running.append(r)

    def run(self) -> List[Request]:
        while self.waiting or self.running:
            self._admit()
            if not self.running:
                continue

            start = time.perf_counter()
            step_batch_incremental(self.model, self.running)
            self.wall_time += time.perf_counter() - start
            self.num_decode_rounds += 1

            still_running = []
            for r in self.running:
                hit_eos = r.generated_ids[-1] == self.eos_token_id
                hit_max = r.num_generated >= r.max_new_tokens
                if hit_eos or hit_max:
                    self._finish(r)
                else:
                    still_running.append(r)
            self.running = still_running

        return self.finished
