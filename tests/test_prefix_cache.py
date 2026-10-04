import torch

from prefix_cache import PrefixCache


def _kv():
    return [(torch.ones(2, 3, 4), torch.zeros(2, 3, 4))]


def test_miss_then_hit_counts():
    c = PrefixCache()
    assert c.get([1, 2, 3]) is None
    c.put([1, 2, 3], _kv())
    assert c.get([1, 2, 3]) is not None
    assert (c.hits, c.misses) == (1, 1)


def test_exact_match_only():
    c = PrefixCache()
    c.put([1, 2, 3], _kv())
    assert c.get([1, 2]) is None
    assert c.get([1, 2, 3, 4]) is None


def test_get_returns_copy_not_alias():
    c = PrefixCache()
    c.put([1], _kv())
    got = c.get([1])
    got[0][0].fill_(7)
    assert torch.equal(c.get([1])[0][0], torch.ones(2, 3, 4))


def test_put_copies_input_and_does_not_overwrite():
    c = PrefixCache()
    kv = _kv()
    c.put([1], kv)
    kv[0][0].fill_(9)
    assert torch.equal(c.get([1])[0][0], torch.ones(2, 3, 4))
    c.put([1], [(torch.full((2, 3, 4), 5.0), torch.zeros(2, 3, 4))])
    assert torch.equal(c.get([1])[0][0], torch.ones(2, 3, 4))
