"""
Milestone 4: prefix caching.

Exact-match cache keyed by a token-id prefix (e.g. a shared system
prompt): the first request to prefill a given prefix pays full cost and
stores its per-layer KV; every later request with the identical prefix
reuses that KV and only prefills its own unique suffix, skipping
attention/FFN compute for the shared tokens entirely.

This is deliberately exact-match rather than a general radix-tree over
arbitrary token boundaries (what vLLM/SGLang do in production) -- the
milestone's stated goal ("requests sharing a system prompt reuse cached
blocks") is a single shared-prefix scenario, and exact-match keyed by
the prefix's token tuple covers it without the added complexity of
partial-prefix matching / eviction policy design.
"""

from typing import Dict, List, Optional, Tuple

import torch

LayerKV = Tuple[torch.Tensor, torch.Tensor]  # (keys, values), each [num_heads, seq_len, head_dim]


class PrefixCache:
    def __init__(self):
        self._store: Dict[Tuple[int, ...], List[LayerKV]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, prefix_ids: List[int]) -> Optional[List[LayerKV]]:
        key = tuple(prefix_ids)
        cached = self._store.get(key)
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        # Clone so the caller's mutations (e.g. growing this KV during
        # decode) never alias the cached copy.
        return [(k.clone(), v.clone()) for k, v in cached]

    def put(self, prefix_ids: List[int], layer_kv: List[LayerKV]) -> None:
        key = tuple(prefix_ids)
        if key in self._store:
            return
        self._store[key] = [(k.clone(), v.clone()) for k, v in layer_kv]
