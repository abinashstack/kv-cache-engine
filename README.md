# kv-cache-engine

A small inference engine that reimplements the memory-management and
scheduling techniques behind production LLM servers (vLLM, SGLang):
paged KV cache storage, continuous batching, and prefix caching —
served over a FastAPI endpoint.

Base model: GPT-2 small (124M), run on CPU.

## Milestones

1. **Baseline** — naive full KV cache via HF `transformers`, measure
   latency/throughput/memory as the reference point. (`benchmarks/baseline.py`)
2. **Paged KV cache** — store KV in fixed-size blocks with a per-sequence
   block table instead of one contiguous tensor, eliminating fragmentation.
3. **Continuous batching scheduler** — iteration-level scheduling: new
   requests join a running batch as soon as a slot frees up.
4. **Prefix caching** — hash/radix-tree the token prefix so requests
   sharing a system prompt reuse cached blocks instead of recomputing.
5. **Serving layer** — FastAPI endpoint with streaming responses.
6. **Benchmarks** — tokens/sec, p50/p99 latency, memory footprint, and
   prefix cache hit rate, plotted against the Milestone 1 baseline.

## Setup

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Status

- [x] Milestone 1: baseline
- [ ] Milestone 2: paged KV cache
- [ ] Milestone 3: continuous batching
- [ ] Milestone 4: prefix caching
- [ ] Milestone 5: serving layer
- [ ] Milestone 6: final benchmarks
