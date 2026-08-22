"""
Milestone 5: live serving scheduler.

scheduler.Scheduler is a run-to-completion batch job: you add a fixed
set of requests up front and call run(). A real server needs the
opposite shape -- requests arrive continuously from arbitrary HTTP
connections and each needs its own stream of tokens as they're
produced, while the scheduler keeps batching whatever's concurrently
active. LiveScheduler adapts the same underlying primitives (prefill,
prefill_with_prefix_cache, step_batch_incremental -- all already
validated in Milestones 2-4) to that shape:

  - submit() is thread-safe and can be called from an asyncio request
    handler; it hands back a LiveRequest whose output_queue receives
    each generated token id as soon as it's produced, terminated by a
    None sentinel.
  - The scheduling loop itself runs on a dedicated background thread
    (torch's CPU forward passes are blocking; running them on the
    asyncio event loop would stall every other in-flight HTTP request
    for the duration of each round).
"""

import itertools
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from prefix_cache import PrefixCache
from scheduler import Request, prefill, prefill_with_prefix_cache, step_batch_incremental

_id_counter = itertools.count()


@dataclass
class LiveRequest:
    request_id: int
    prompt_ids: List[int]
    max_new_tokens: int
    shared_prefix_len: int = 0
    output_queue: "queue.Queue" = field(default_factory=queue.Queue)


class LiveScheduler:
    def __init__(
        self,
        model,
        eos_token_id: int,
        max_batch_size: int = 4,
        prefix_cache: Optional[PrefixCache] = None,
        poll_interval: float = 0.005,
    ):
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_batch_size = max_batch_size
        self.prefix_cache = prefix_cache
        self.poll_interval = poll_interval

        self._incoming: "queue.Queue" = queue.Queue()
        self._waiting: List[Request] = []
        self._running: List[Request] = []
        self._live_by_id = {}

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def submit(self, prompt_ids: List[int], max_new_tokens: int, shared_prefix_len: int = 0) -> LiveRequest:
        request_id = next(_id_counter)
        live_req = LiveRequest(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            shared_prefix_len=shared_prefix_len,
        )
        self._incoming.put(live_req)
        return live_req

    def _emit(self, r: Request, done: bool) -> None:
        live_req = self._live_by_id[r.request_id]
        live_req.output_queue.put(r.generated_ids[-1])
        if done:
            live_req.output_queue.put(None)
            del self._live_by_id[r.request_id]

    def _drain_incoming(self) -> None:
        while True:
            try:
                live_req = self._incoming.get_nowait()
            except queue.Empty:
                return
            sched_req = Request(
                request_id=live_req.request_id,
                prompt_ids=live_req.prompt_ids,
                max_new_tokens=live_req.max_new_tokens,
                shared_prefix_len=live_req.shared_prefix_len,
            )
            self._live_by_id[live_req.request_id] = live_req
            self._waiting.append(sched_req)

    def _admit(self) -> None:
        slots = self.max_batch_size - len(self._running)
        if slots <= 0 or not self._waiting:
            return
        to_admit = self._waiting[:slots]
        del self._waiting[:slots]

        for r in to_admit:
            if self.prefix_cache is not None and r.shared_prefix_len > 0:
                prefill_with_prefix_cache(self.model, r, self.prefix_cache, r.shared_prefix_len)
            else:
                prefill(self.model, r)

            done = r.generated_ids[-1] == self.eos_token_id or r.num_generated >= r.max_new_tokens
            self._emit(r, done)
            if not done:
                self._running.append(r)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain_incoming()
            self._admit()

            if not self._running:
                time.sleep(self.poll_interval)
                continue

            step_batch_incremental(self.model, self._running)

            still_running = []
            for r in self._running:
                done = r.generated_ids[-1] == self.eos_token_id or r.num_generated >= r.max_new_tokens
                self._emit(r, done)
                if not done:
                    still_running.append(r)
            self._running = still_running
