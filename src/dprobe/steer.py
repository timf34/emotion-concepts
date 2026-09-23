"""Run the extended elicitation under activation steering and judge the result.

Two backends:
  hf        : forward hooks add coeff * unit_vector to the residual stream at the chosen layers
              during model.generate (guaranteed to work; slow, batch across conversations per turn).
  easysteer : ZJU-REAL/EasySteer (vLLM 0.29 fork). Vectors are read from the gguf files written
              by extract.export_gguf; steering is passed per request via SteeringSpec.

Steering strength is expressed as a fraction of the typical residual norm at that layer
(Anthropic: -0.1 .. +0.1), converted to an absolute coefficient using the neutral-token norm
recorded at extraction time. Output transcripts: RESULTS_DIR/spiral/<model>/extended_steer-<label>@<layers>x<strength>/
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch

from dprobe.config import RESULTS_DIR, SpiralConfig, get_model
from dprobe.extract import load_vectors, vectors_dir
from dprobe.models import add_vectors, load_model
from dprobe.spiral import run_extended
from dprobe.transcripts import render


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-6)


def residual_norms(model_key: str) -> dict[int, float]:
    """Mean neutral-token residual norm per analysis layer (recorded at extraction; falls back to the mean vector's norm)."""
    pca = torch.load(vectors_dir(model_key) / "neutral_pca.pt")
    return {l: float(d.get("resid_norm") or d["mean"].norm()) for l, d in pca.items()}


def steering_vectors(model_key: str, label: str, layers: list[int], strength: float, set_name: str | None = None) -> dict[int, torch.Tensor]:
    """coeff_l = strength * ||resid_l|| ; vector = coeff_l * unit(v_l)."""
    if set_name is None:
        set_name = "syndromes" if label in ("clinical_depression", "acute_grief", "burnout_exhaustion", "physical_illness_fatigue", "anxiety_panic", "frustration_blocked_goal") else "emotions"
    vecs = load_vectors(model_key, set_name, denoised=True)[label]
    norms = residual_norms(model_key)
    return {l: strength * norms[l] * _unit(vecs[l]) for l in layers}


def tag_for(label: str, layers: list[int], strength: float) -> str:
    ls = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])
    return f"steer-{label.replace(' ', '_')}@{ls}x{strength:+g}"


# ---------------------------------------------------------------------------
# HF hooks backend
# ---------------------------------------------------------------------------
def run_steered_hf(model_key: str, label: str, layers: list[int], strength: float, cfg: SpiralConfig | None = None, positions: str = "all", batch: int = 8, model_bundle=None) -> Path:
    cfg = cfg or SpiralConfig()
    model, tok, spec = model_bundle or load_model(model_key)
    vecs = steering_vectors(model_key, label, layers, strength)
    device = next(model.parameters()).device
    tok.padding_side = "left"
    tag = tag_for(label, layers, strength)
    queue: list[tuple[list[dict], asyncio.Future]] = []
    loop_holder: dict = {}

    async def generate_fn(messages, sample):
        fut = loop_holder["loop"].create_future()
        queue.append((messages, fut))
        return await fut

    @torch.no_grad()
    def flush():
        while queue:
            chunk, queue[:] = queue[:batch], queue[batch:]
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m, _ in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            plen = enc["input_ids"].shape[1]
            with add_vectors(model, vecs, positions=positions, prompt_len=plen):
                out = model.generate(**enc, max_new_tokens=cfg.max_tokens, do_sample=True, temperature=cfg.temperature, pad_token_id=tok.pad_token_id)
            for (m, fut), row in zip(chunk, out):
                fut.set_result(tok.decode(row[plen:], skip_special_tokens=True))

    # Drive the async conversation runner with a batching executor: run_extended awaits generate_fn,
    # we periodically flush the queue on the same loop.
    async def driver():
        loop_holder["loop"] = asyncio.get_running_loop()
        task = asyncio.get_running_loop().run_in_executor(None, lambda: None)
        runner = asyncio.create_task(asyncio.to_thread(run_extended, model_key, cfg, generate_fn, tag))
        # run_extended calls asyncio.run internally in a thread; instead we flush in this loop
        while not runner.done():
            await asyncio.sleep(0.05)
            if queue:
                flush()
        await runner
    # Simpler and robust: monkey-patch dprobe.openrouter.run to use our loop
    import dprobe.spiral as sp

    def _run(coro):
        async def wrapped():
            loop_holder["loop"] = asyncio.get_running_loop()
            main = asyncio.create_task(coro)
            while not main.done():
                await asyncio.sleep(0.02)
                if queue:
                    flush()
            return main.result()
        return asyncio.run(wrapped())

    orig = sp.run
    sp.run = _run
    try:
        out = run_extended(model_key, cfg, generate_fn, tag)
    finally:
        sp.run = orig
    return out


# ---------------------------------------------------------------------------
# EasySteer backend
# ---------------------------------------------------------------------------
def run_steered_easysteer(model_key: str, label: str, layers: list[int], strength: float, cfg: SpiralConfig | None = None, gpu_mem: float = 0.9) -> Path:
    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    cfg = cfg or SpiralConfig()
    spec = get_model(model_key)
    set_name = "syndromes" if label.startswith(("clinical", "acute", "burnout", "physical", "anxiety_", "frustration_")) else "emotions"
    gguf_path = vectors_dir(model_key) / "gguf" / set_name / f"{label.replace(' ', '_')}.gguf"
    if not gguf_path.exists():
        raise FileNotFoundError(gguf_path)
    norms = residual_norms(model_key)
    vecs = load_vectors(model_key, set_name, denoised=True)[label]
    # EasySteer scales the stored vector; store-time vectors are raw difference-of-means, so convert:
    # coeff_l = strength * ||resid_l|| / ||v_l||  -> use the mean over layers as a single scale (EasySteer takes one scale).
    scale = float(torch.tensor([strength * norms[l] / (vecs[l].norm() + 1e-6) for l in layers]).mean())
    llm = LLM(model=spec.hf_id, dtype="bfloat16", gpu_memory_utilization=gpu_mem, enable_steer_vector=True, steer_algorithms=["direct"], max_model_len=20000)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    steering = SteeringSpec(vectors=[VectorSpec(source=str(gguf_path), scale=scale, layers=layers, apply=ApplySpec(prompt="all", generation="all"))])
    tag = tag_for(label, layers, strength)

    queue: list[tuple[list[dict], asyncio.Future]] = []

    async def generate_fn(messages, sample):
        fut = asyncio.get_running_loop().create_future()
        queue.append((messages, fut))
        return await fut

    def flush():
        chunk, queue[:] = queue[:], []
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m, _ in chunk]
        outs = llm.generate(prompts, sampling_params=sp, steering=steering)
        for (_, fut), o in zip(chunk, outs):
            fut.set_result(o.outputs[0].text)

    import dprobe.spiral as spmod

    def _run(coro):
        async def wrapped():
            main = asyncio.create_task(coro)
            while not main.done():
                await asyncio.sleep(0.05)
                if queue and len(queue) >= min(cfg.max_concurrency, 64):
                    flush()
                elif queue and all(t.done() for t in asyncio.all_tasks() if t is not main and t is not asyncio.current_task()):
                    flush()
            return main.result()
        return asyncio.run(wrapped())

    orig = spmod.run
    spmod.run = _run
    try:
        out = run_extended(model_key, cfg, generate_fn, tag)
    finally:
        spmod.run = orig
    return out


def run_steering_grid(model_key: str, labels: list[str], strengths: list[float], layers: list[int], backend: str = "hf", rollouts: int = 40, judge: bool = True):
    cfg = SpiralConfig(rollouts=rollouts, extra_puzzle_rollouts=0)
    outs = []
    bundle = load_model(model_key) if backend == "hf" else None
    for label in labels:
        for s in strengths:
            if backend == "hf":
                p = run_steered_hf(model_key, label, layers, s, cfg, model_bundle=bundle)
            else:
                p = run_steered_easysteer(model_key, label, layers, s, cfg)
            outs.append(p)
    if judge:
        from dprobe.judge import judge_transcripts, summarize
        for p in outs:
            judge_transcripts(p, "frustration")
            print(p.parent.name, json.dumps(summarize(p)))
    return outs
