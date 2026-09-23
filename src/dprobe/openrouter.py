"""Async OpenRouter chat client with per-request disk cache, retries and concurrency control.

Every request is cached individually (hash of model + messages + params + sample index),
so multi-turn conversations and large story batches are resumable after interruption.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI

from dprobe.config import CACHE_DIR, OPENROUTER_BASE_URL

MAX_RETRIES = 5
Messages = list[dict[str, str]]


def _key(model: str, messages: Messages, params: dict, sample: int) -> str:
    payload = json.dumps({"m": model, "msgs": messages, "p": params, "s": sample}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class OpenRouterClient:
    def __init__(self, cache_dir: Path | None = None, max_concurrency: int = 32, app_name: str = "dprobe"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (see .env.example)")
        self._api_key = api_key
        self._app_name = app_name
        self._clients: dict[int, AsyncOpenAI] = {}
        self._sems: dict[int, asyncio.Semaphore] = {}
        self.max_concurrency = max_concurrency
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hit": 0, "miss": 0, "fail": 0}

    # One AsyncOpenAI + one semaphore per running event loop, so a client survives repeated asyncio.run().
    def _loop_state(self):
        lid = id(asyncio.get_running_loop())
        if lid not in self._clients:
            self._clients[lid] = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self._api_key,
                default_headers={"HTTP-Referer": "https://github.com/timf34", "X-Title": self._app_name},
                timeout=300.0,
            )
            self._sems[lid] = asyncio.Semaphore(self.max_concurrency)
        return self._clients[lid], self._sems[lid]

    # ------------------------------------------------------------------
    async def chat(
        self,
        model: str,
        messages: Messages,
        *,
        sample: int = 0,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        extra_body: dict | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Return {"content": str|None, "finish_reason", "usage", "provider", "cached": bool}."""
        params = {"temperature": temperature, "max_tokens": max_tokens, "extra_body": extra_body or {}}
        path = self.cache_dir / model.replace("/", "__") / f"{_key(model, messages, params, sample)}.json"
        if path.exists() and not force:
            with open(path) as f:
                out = json.load(f)
            if out.get("content"):
                self.stats["hit"] += 1
                out["cached"] = True
                return out

        client, sem = self._loop_state()
        async with sem:
            out = await self._call(client, model, messages, temperature, max_tokens, extra_body or {})
        if out.get("content"):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(out, f)
            os.replace(tmp, path)
            self.stats["miss"] += 1
        else:
            self.stats["fail"] += 1
        out["cached"] = False
        return out

    async def _call(self, client, model, messages, temperature, max_tokens, extra_body) -> dict[str, Any]:
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                choice = resp.choices[0]
                content = choice.message.content
                if not content or not content.strip():
                    raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
                usage = resp.usage.model_dump() if resp.usage else None
                return {
                    "content": content,
                    "finish_reason": choice.finish_reason,
                    "usage": usage,
                    "provider": getattr(resp, "provider", None),
                    "model": resp.model,
                    "ts": time.time(),
                }
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(60, (2 ** attempt) + random.random())
                print(f"[openrouter] {model} attempt {attempt + 1}/{MAX_RETRIES} failed: {type(e).__name__}: {str(e)[:160]} (retry in {wait:.0f}s)")
                await asyncio.sleep(wait)
        return {"content": None, "error": f"{type(last_err).__name__}: {last_err}"}

    # ------------------------------------------------------------------
    async def batch(
        self,
        model: str,
        message_lists: Sequence[Messages],
        *,
        n: int = 1,
        progress_every: int = 50,
        **kw,
    ) -> list[list[str | None]]:
        """n completions for each message list. Returns [[content or None] * n] per input."""
        total = len(message_lists) * n
        done = 0
        t0 = time.time()

        async def one(i: int, s: int):
            nonlocal done
            out = await self.chat(model, message_lists[i], sample=s, **kw)
            done += 1
            if done % progress_every == 0 or done == total:
                el = time.time() - t0
                print(f"[openrouter] {model}: {done}/{total} done ({el:.0f}s) hits={self.stats['hit']} miss={self.stats['miss']} fail={self.stats['fail']}")
            return i, s, out.get("content")

        results = await asyncio.gather(*[one(i, s) for i in range(len(message_lists)) for s in range(n)])
        grid: list[list[str | None]] = [[None] * n for _ in message_lists]
        for i, s, c in results:
            grid[i][s] = c
        return grid


def run(coro):
    """asyncio.run that also works inside notebooks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import nest_asyncio  # type: ignore

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)
