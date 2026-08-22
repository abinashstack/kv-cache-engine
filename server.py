"""
Milestone 5: FastAPI serving layer with streaming responses.

Wraps LiveScheduler (serving.py) so multiple concurrent HTTP requests
are transparently continuous-batched together on the model, exactly as
validated in Milestone 3, with optional prefix-cache reuse (Milestone 4)
via an explicit system_prompt field shared across requests.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from prefix_cache import PrefixCache
from serving import LiveScheduler

MODEL_NAME = "gpt2"
MAX_BATCH_SIZE = 4

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    prefix_cache = PrefixCache()
    scheduler = LiveScheduler(
        model,
        eos_token_id=model.config.eos_token_id,
        max_batch_size=MAX_BATCH_SIZE,
        prefix_cache=prefix_cache,
    )
    scheduler.start()

    state["tokenizer"] = tokenizer
    state["scheduler"] = scheduler
    state["prefix_cache"] = prefix_cache

    yield

    scheduler.stop()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    system_prompt: Optional[str] = None


@app.post("/generate")
async def generate(req: GenerateRequest):
    tokenizer = state["tokenizer"]
    scheduler: LiveScheduler = state["scheduler"]

    shared_prefix_len = 0
    if req.system_prompt:
        prefix_ids = tokenizer(req.system_prompt, return_tensors="pt").input_ids[0].tolist()
        suffix_ids = tokenizer(req.prompt, return_tensors="pt").input_ids[0].tolist()
        prompt_ids = prefix_ids + suffix_ids
        shared_prefix_len = len(prefix_ids)
    else:
        prompt_ids = tokenizer(req.prompt, return_tensors="pt").input_ids[0].tolist()

    live_req = scheduler.submit(prompt_ids, req.max_new_tokens, shared_prefix_len=shared_prefix_len)

    async def token_stream():
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, live_req.output_queue.get)
            if token is None:
                break
            yield tokenizer.decode([token])

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok"}
