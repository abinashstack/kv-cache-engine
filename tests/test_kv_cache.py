import pytest
import torch

from kv_cache import BlockAllocator, BlockAllocatorConfig, PagedKVCache

CFG = BlockAllocatorConfig(block_size=4, num_blocks=8, num_layers=2, num_heads=2, head_dim=3)


def _kv(n, start=0):
    base = torch.arange(start, start + n, dtype=torch.float32).view(n, 1, 1)
    t = base.expand(n, CFG.num_heads, CFG.head_dim).clone()
    return [t.clone() for _ in range(CFG.num_layers)], [t.clone() + 1000 for _ in range(CFG.num_layers)]


def test_allocator_alloc_and_free():
    a = BlockAllocator(CFG)
    ids = a.allocate(3)
    assert len(set(ids)) == 3 and len(a.free_blocks) == 5
    a.free(ids)
    assert len(a.free_blocks) == 8


def test_allocator_exhaustion_raises():
    a = BlockAllocator(CFG)
    a.allocate(8)
    with pytest.raises(RuntimeError, match="Insufficient blocks"):
        a.allocate(1)


def test_roundtrip_across_block_boundaries():
    cache = PagedKVCache(CFG)
    seq = cache.new_sequence()
    k, v = _kv(10)  # 10 tokens -> 3 blocks of 4
    cache.add_kv(seq, k, v)
    assert len(cache.sequences[seq].blocks) == 3
    for layer, (gk, gv) in enumerate(cache.get_kv(seq)):
        assert torch.equal(gk, k[layer])
        assert torch.equal(gv, v[layer])


def test_incremental_append_matches_single_write():
    cache = PagedKVCache(CFG)
    seq = cache.new_sequence()
    for i in range(9):
        k, v = _kv(1, start=i)
        cache.add_kv(seq, k, v)
    full_k, _ = _kv(9)
    assert cache.sequences[seq].token_count == 9
    assert torch.equal(cache.get_kv(seq)[0][0], full_k[0])


def test_sequences_are_isolated():
    cache = PagedKVCache(CFG)
    a, b = cache.new_sequence(), cache.new_sequence()
    ka, va = _kv(5, start=0)
    kb, vb = _kv(6, start=100)
    cache.add_kv(a, ka, va)
    cache.add_kv(b, kb, vb)
    assert torch.equal(cache.get_kv(a)[1][0], ka[1])
    assert torch.equal(cache.get_kv(b)[1][0], kb[1])


def test_free_sequence_returns_blocks():
    cache = PagedKVCache(CFG)
    seq = cache.new_sequence()
    cache.add_kv(seq, *_kv(10))
    assert cache.memory_usage()["used_blocks"] == 3
    cache.free_sequence(seq)
    assert cache.memory_usage()["used_blocks"] == 0
    cache.free_sequence(seq)  # idempotent


def test_memory_usage_bytes():
    cache = PagedKVCache(CFG)
    cache.add_kv(cache.new_sequence(), *_kv(4))
    block_bytes = 2 * 4 * 2 * 3 * 4 * 2
    assert cache.memory_usage()["used_cache_mb"] == pytest.approx(block_bytes / 2**20)
