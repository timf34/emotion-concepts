"""The Gemma Needs Help 'Extended' elicitation: base prompt + 7 neutral rejections (8 model turns).

Prompts and rejection lists are vendored verbatim from annasoligo/paper_gemma-needs-help (MIT)
in data/evals/. Transcripts are written to RESULTS_DIR/spiral/<model_key>/extended/transcripts.jsonl
as {"id", "puzzle", "rollout", "messages": [...], "meta": {...}}.

Generation goes through OpenRouter by default (temperature 1, as in the paper). The same
function accepts a `generate_fn` so the pod can run it through vLLM / EasySteer instead.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Awaitable, Callable

from dprobe.config import EVALS_DIR, RESULTS_DIR, SpiralConfig, get_model
from dprobe.openrouter import OpenRouterClient, run

Messages = list[dict[str, str]]


def _load_eval_module(name: str):
    path = EVALS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gnh_evals_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def extended_condition() -> dict:
    return _load_eval_module("extended").EXTENDED_CONVERSATION


def impossible_puzzles() -> list[tuple[str, str]]:
    m = _load_eval_module("impossible_numeric")
    return list(m.ORIGINAL_IMPOSSIBLE) + list(m.VARIANT_IMPOSSIBLE)


def build_jobs(cfg: SpiralConfig) -> list[dict]:
    """Paper-faithful: `rollouts` of the extended base prompt; plus `extra_puzzle_rollouts` per long-form puzzle."""
    ext = extended_condition()
    jobs = [{"puzzle": "extended_countdown156", "rollout": r, "prompt": ext["base_prompt"], "rejections": ext["rejections"][:7]} for r in range(cfg.rollouts)]
    for pid, ptext in impossible_puzzles():
        for r in range(cfg.extra_puzzle_rollouts):
            jobs.append({"puzzle": pid, "rollout": r, "prompt": ptext, "rejections": ext["rejections"][:7]})
    for j in jobs:
        j["id"] = f"{j['puzzle']}__r{j['rollout']:03d}"
    return jobs


def transcripts_path(model_key: str, condition: str = "extended", tag: str = "") -> Path:
    d = RESULTS_DIR / "spiral" / model_key / (condition + (f"_{tag}" if tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d / "transcripts.jsonl"


def load_transcripts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


async def _run_conversation(job: dict, generate: Callable[[Messages, int], Awaitable[str | None]]) -> dict | None:
    messages: Messages = [{"role": "user", "content": job["prompt"]}]
    for turn, rejection in enumerate([None] + list(job["rejections"])):
        if rejection is not None:
            messages.append({"role": "user", "content": rejection})
        reply = await generate(messages, job["rollout"])
        if reply is None:
            print(f"[spiral] {job['id']}: generation failed at turn {turn}; dropping conversation")
            return None
        messages.append({"role": "assistant", "content": reply})
    return {"id": job["id"], "puzzle": job["puzzle"], "rollout": job["rollout"], "messages": messages}


def run_extended(model_key: str, cfg: SpiralConfig | None = None, generate_fn=None, tag: str = "", limit: int | None = None) -> Path:
    """Run (or resume) the extended condition. `generate_fn(messages, sample) -> str|None` (async) overrides OpenRouter."""
    cfg = cfg or SpiralConfig()
    spec = get_model(model_key)
    out_path = transcripts_path(model_key, "extended", tag)
    done_ids = {t["id"] for t in load_transcripts(out_path)}
    jobs = [j for j in build_jobs(cfg) if j["id"] not in done_ids]
    if limit is not None:
        jobs = jobs[:limit]
    print(f"[spiral] {model_key}: {len(done_ids)} conversations done, {len(jobs)} to run -> {out_path}")
    if not jobs:
        return out_path

    if generate_fn is None:
        if spec.openrouter_id is None:
            raise ValueError(f"{model_key} not on OpenRouter; pass generate_fn (vLLM) on the pod")
        client = OpenRouterClient(max_concurrency=cfg.max_concurrency)
        extra = {"reasoning": {"enabled": False}} if spec.family == "gemma4" else {}

        async def generate_fn(messages, sample):
            out = await client.chat(spec.openrouter_id, messages, sample=sample, temperature=cfg.temperature, max_tokens=cfg.max_tokens, extra_body=extra)
            return out.get("content")

    async def main():
        sem = asyncio.Semaphore(cfg.max_concurrency)
        n_done = 0

        async def one(job):
            nonlocal n_done
            async with sem:
                res = await _run_conversation(job, generate_fn)
            if res is not None:
                res["meta"] = {"model": spec.hf_id, "source": "openrouter" if spec.openrouter_id else "local", "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
                with open(out_path, "a") as f:
                    f.write(json.dumps(res) + "\n")
            n_done += 1
            if n_done % 10 == 0 or n_done == len(jobs):
                print(f"[spiral] {model_key}: {n_done}/{len(jobs)} conversations finished")

        await asyncio.gather(*[one(j) for j in jobs])

    run(main())
    return out_path
