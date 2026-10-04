"""LiveScheduler robustness tests. The model primitives are replaced with
fakes, so no model weights are needed."""

import queue

import pytest

import serving
from serving import LiveScheduler, StreamError

EOS = 99


def fake_prefill(model, r):
    r.generated_ids.append(1)


def fake_step(model, reqs):
    for r in reqs:
        r.generated_ids.append(len(r.generated_ids) + 1)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(serving, "prefill", fake_prefill)
    monkeypatch.setattr(serving, "prefill_with_prefix_cache", lambda m, r, c, n: fake_prefill(m, r))
    monkeypatch.setattr(serving, "step_batch_incremental", fake_step)


def collect(live, timeout=5):
    """Drain a request's stream; returns (tokens, errors). Fails the test on a hang."""
    tokens, errors = [], []
    while True:
        try:
            item = live.output_queue.get(timeout=timeout)
        except queue.Empty:
            pytest.fail("stream hung: no sentinel received")
        if item is None:
            return tokens, errors
        (errors if isinstance(item, StreamError) else tokens).append(item)


@pytest.fixture
def sched(patched):
    s = LiveScheduler(model=None, eos_token_id=EOS, max_batch_size=2, max_positions=50, poll_interval=0.001)
    s.start()
    yield s
    s.stop()


def test_streams_tokens_then_sentinel(sched):
    live = sched.submit([5, 6], max_new_tokens=4)
    tokens, errors = collect(live)
    assert tokens == [1, 2, 3, 4] and not errors


def test_more_requests_than_slots_all_finish(sched):
    reqs = [sched.submit([5], max_new_tokens=3) for _ in range(5)]
    for r in reqs:
        assert collect(r)[0] == [1, 2, 3]


@pytest.mark.parametrize(
    "prompt,max_new,prefix",
    [
        ([], 5, 0),  # empty prompt
        ([1, 2], 0, 0),  # nothing to generate
        ([1, 2], -3, 0),
        ([1, 2], 5, 2),  # empty suffix after the shared prefix
        ([1, 2], 5, -1),
        ([1] * 40, 11, 0),  # 51 > max_positions (50)
    ],
)
def test_invalid_requests_rejected_up_front(sched, prompt, max_new, prefix):
    with pytest.raises(ValueError):
        sched.submit(prompt, max_new, shared_prefix_len=prefix)


def test_request_at_context_limit_is_accepted(sched):
    live = sched.submit([1] * 40, 10)  # exactly 50
    assert len(collect(live)[0]) == 10


def test_prefill_failure_is_isolated(monkeypatch, patched):
    def flaky_prefill(model, r):
        if r.prompt_ids == [666]:
            raise RuntimeError("boom")
        fake_prefill(model, r)

    monkeypatch.setattr(serving, "prefill", flaky_prefill)
    s = LiveScheduler(None, EOS, max_batch_size=2, poll_interval=0.001)
    s.start()
    try:
        bad = s.submit([666], 3)
        good = s.submit([1], 3)
        tokens, errors = collect(bad)
        assert tokens == [] and "boom" in errors[0].message
        assert collect(good)[0] == [1, 2, 3]
        # the thread must still be serving afterwards
        assert collect(s.submit([2], 2))[0] == [1, 2]
    finally:
        s.stop()


def test_decode_failure_fails_batch_but_loop_survives(monkeypatch, patched):
    calls = {"n": 0}

    def flaky_step(model, reqs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("step exploded")
        fake_step(model, reqs)

    monkeypatch.setattr(serving, "step_batch_incremental", flaky_step)
    s = LiveScheduler(None, EOS, max_batch_size=2, poll_interval=0.001)
    s.start()
    try:
        first = s.submit([1], 5)
        _, errors = collect(first)
        assert errors and "step exploded" in errors[0].message
        assert collect(s.submit([1], 3))[0] == [1, 2, 3]
        assert s._thread.is_alive()
    finally:
        s.stop()


def test_stop_releases_in_flight_clients(monkeypatch, patched):
    monkeypatch.setattr(serving, "step_batch_incremental", lambda m, r: [fake_step(m, r), __import__("time").sleep(0.01)])
    s = LiveScheduler(None, EOS, max_batch_size=1, poll_interval=0.001)
    s.start()
    running = s.submit([1], 10_000)
    waiting = s.submit([1], 10_000)  # stuck behind the single slot
    s.stop()
    for live in (running, waiting):
        _, errors = collect(live, timeout=2)
        assert errors and "shutting down" in errors[-1].message


def test_submit_after_stop_rejected(patched):
    s = LiveScheduler(None, EOS, poll_interval=0.001)
    s.start()
    s.stop()
    with pytest.raises(RuntimeError):
        s.submit([1], 1)
