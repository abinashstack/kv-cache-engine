"""HTTP-layer tests with a fake tokenizer and fake model primitives."""

import pytest
import torch
from fastapi.testclient import TestClient

import server
import serving
from serving import LiveScheduler


class FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % 90 + 1 for c in text]
        return type("Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(serving, "prefill", lambda m, r: r.generated_ids.append(1))
    monkeypatch.setattr(serving, "prefill_with_prefix_cache", lambda m, r, c, n: r.generated_ids.append(1))
    monkeypatch.setattr(
        serving, "step_batch_incremental", lambda m, rs: [r.generated_ids.append(2) for r in rs]
    )
    sched = LiveScheduler(None, eos_token_id=99, max_batch_size=2, max_positions=64, poll_interval=0.001)
    sched.start()
    server.state.update(tokenizer=FakeTokenizer(), scheduler=sched)
    yield TestClient(server.app)  # no `with`: skips the real model-loading lifespan
    sched.stop()
    server.state.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_generate_streams_text(client):
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 3})
    assert r.status_code == 200 and len(r.text) == 3


@pytest.mark.parametrize("n", [0, -1, 5000])
def test_max_new_tokens_bounds_rejected(client, n):
    assert client.post("/generate", json={"prompt": "hi", "max_new_tokens": n}).status_code == 422


def test_empty_prompt_is_400(client):
    assert client.post("/generate", json={"prompt": ""}).status_code == 400


def test_system_prompt_with_empty_prompt_is_400(client):
    r = client.post("/generate", json={"prompt": "", "system_prompt": "sys"})
    assert r.status_code == 400


def test_context_overflow_is_400(client):
    r = client.post("/generate", json={"prompt": "x" * 60, "max_new_tokens": 10})
    assert r.status_code == 400 and "context window" in r.json()["detail"]


def test_shutting_down_is_503(client):
    server.state["scheduler"].stop()
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 503
