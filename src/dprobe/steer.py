"""Run the extended elicitation under activation steering and judge the result.

Two backends:
  hf        : forward hooks add a scaled unit vector to the residual stream at the chosen layers during
              model.generate (reference implementation; batches conversations per turn; slow-ish).
  easysteer : ZJU-REAL/EasySteer (vLLM 0.29 fork). Vectors come from the gguf files written by
              extract.export_gguf; steering is passed per request via SteeringSpec. Much faster.

Steering strength is a fraction of the typical neutral-token residual norm at each layer (Anthropic used
-0.1 .. +0.1), converted to an absolute coefficient with the norm recorded at extraction time.

Both backends plug into spiral.run_extended through `generate_fn`: conversations are driven as async
tasks; every pending generation request is queued and the queue is flushed as one batched generate call.
Output: RESULTS_DIR/spiral/<model>/extended_steer-<label>@<layers>x<strength>/transcripts.jsonl (+ judgments).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import torch

import dprobe.spiral as spiral_mod
from dprobe.config import RESULTS_DIR, SYNDROMES, SpiralConfig, get_model
from dprobe.extract import load_vectors, vectors_dir
from dprobe.models import add_vectors, load_model
from dprobe.spiral import run_extended


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-6)


def residual_norms(model_key: str) -> dict[int, float]:
    """Mean neutral-token residual norm per analysis layer (recorded at extraction)."""
    pca = torch.load(vectors_dir(model_key) / "neutral_pca.pt")
    return {l: float(d.get("resid_norm") or d["mean"].norm()) for l, d in pca.items()}


def _set_for(label: str) -> str:
    return "syndromes" if label in SYNDROMES else "emotions"


def steering_vectors(model_key: str, label: str, layers: list[int], strength: float) -> dict[int, torch.Tensor]:
    """{layer: strength * ||resid_l|| * unit(v_l)}"""
    vecs = load_vectors(model_key, _set_for(label), denoised=True)[label]
    norms = residual_norms(model_key)
    missing = [l for l in layers if l not in vecs or l not in norms]
    if missing:
        raise ValueError(f"no denoised vector / norm at layers {missing} (available: {sorted(set(vecs) & set(norms))})")
    return {l: strength * norms[l] * _unit(vecs[l]) for l in layers}


def tag_for(label: str, layers: list[int], strength: float) -> str:
    ls = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])
    return f"steer-{label.replace(' ', '_')}@{ls}x{strength:+g}"


# ---------------------------------------------------------------------------
# Batching driver shared by both backends
# ---------------------------------------------------------------------------
class _BatchDriver:
    """Collects generate requests from the async conversation runner and flushes them in batches."""

    def __init__(self, flush_fn, batch: int, settle_s: float = 0.3):
        self.flush_fn = flush_fn        # (list[messages]) -> list[str]
        self.batch = batch
        self.settle_s = settle_s
        self.queue: list[tuple[list[dict], asyncio.Future]] = []
        self.n_generated = 0

    async def generate(self, messages, sample):
        fut = asyncio.get_running_loop().create_future()
        self.queue.append((messages, fut))
        return await fut

    def _flush(self):
        chunk, self.queue[:] = self.queue[: self.batch], self.queue[self.batch :]
        try:
            outs = self.flush_fn([m for m, _ in chunk])
        except Exception as e:  # noqa: BLE001
            for _, fut in chunk:
                fut.set_exception(e)
            return
        for (_, fut), text in zip(chunk, outs):
            fut.set_result(text)
        self.n_generated += len(chunk)

    def run(self, coro):
        """Replacement for spiral.run: drives `coro` while flushing the queue in batches."""

        async def wrapped():
            main = asyncio.create_task(coro)
            last_len, last_change = 0, time.monotonic()
            while not main.done():
                await asyncio.sleep(0.02)
                n = len(self.queue)
                if n != last_len:
                    last_len, last_change = n, time.monotonic()
                # flush when a full batch is waiting, or when the queue has stopped growing (tail of a turn)
                if n >= self.batch or (n > 0 and time.monotonic() - last_change > self.settle_s):
                    self._flush()
                    last_len, last_change = len(self.queue), time.monotonic()
            return main.result()

        return asyncio.run(wrapped())


def _run_with_driver(model_key: str, cfg: SpiralConfig, driver: _BatchDriver, tag: str) -> Path:
    orig = spiral_mod.run
    spiral_mod.run = driver.run
    try:
        return run_extended(model_key, cfg, driver.generate, tag)
    finally:
        spiral_mod.run = orig


# ---------------------------------------------------------------------------
# HF hooks backend
# ---------------------------------------------------------------------------
def run_steered_hf(model_key: str, label: str, layers: list[int], strength: float, cfg: SpiralConfig | None = None,
                   positions: str = "all", batch: int = 8, model_bundle=None) -> Path:
    cfg = cfg or SpiralConfig()
    model, tok, spec = model_bundle or load_model(model_key)
    vecs = steering_vectors(model_key, label, layers, strength) if strength != 0 else {}
    device = next(model.parameters()).device
    tag = tag_for(label, layers, strength)

    @torch.no_grad()
    def flush(message_lists):
        tok.padding_side = "left"
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = enc["input_ids"].shape[1]
        with add_vectors(model, vecs, positions=positions, prompt_len=plen):
            out = model.generate(**enc, max_new_tokens=cfg.max_tokens, do_sample=True, temperature=cfg.temperature,
                                 pad_token_id=tok.pad_token_id)
        return [tok.decode(row[plen:], skip_special_tokens=True) for row in out]

    print(f"[steer:hf] {model_key} {tag}: {cfg.rollouts} rollouts, layers {layers}, batch {batch}")
    return _run_with_driver(model_key, cfg, _BatchDriver(flush, batch), tag)


# ---------------------------------------------------------------------------
# EasySteer backend
# ---------------------------------------------------------------------------
_EASYSTEER_LLM = {}


def _easysteer_llm(hf_id: str, gpu_mem: float, max_model_len: int):
    from vllm import LLM

    key = (hf_id, gpu_mem, max_model_len)
    if key not in _EASYSTEER_LLM:
        _EASYSTEER_LLM[key] = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=gpu_mem, enable_steer_vector=True,
                                  steer_algorithms=["direct"], max_model_len=max_model_len)
    return _EASYSTEER_LLM[key]


def run_steered_easysteer(model_key: str, label: str, layers: list[int], strength: float, cfg: SpiralConfig | None = None,
                          gpu_mem: float = 0.9, max_model_len: int = 20000, batch: int = 64) -> Path:
    from vllm import SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    cfg = cfg or SpiralConfig()
    spec = get_model(model_key)
    tag = tag_for(label, layers, strength)
    llm = _easysteer_llm(spec.hf_id, gpu_mem, max_model_len)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens)

    steering = False
    if strength != 0:
        gguf_path = vectors_dir(model_key) / "gguf" / _set_for(label) / f"{label.replace(' ', '_')}.gguf"
        if not gguf_path.exists():
            raise FileNotFoundError(gguf_path)
        # gguf holds the raw denoised difference-of-means; EasySteer applies one scalar. Convert the per-layer
        # target (strength * ||resid_l|| along unit(v_l)) into that scalar using the mean over the chosen layers.
        vecs = load_vectors(model_key, _set_for(label), denoised=True)[label]
        norms = residual_norms(model_key)
        scale = float(torch.tensor([strength * norms[l] / (vecs[l].norm() + 1e-6) for l in layers]).mean())
        steering = SteeringSpec(vectors=[VectorSpec(source=str(gguf_path), scale=scale, layers=list(layers),
                                                    apply=ApplySpec(prompt="all", generation="all"))])

    def flush(message_lists):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
        outs = llm.generate(prompts, sampling_params=sp, steering=steering, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    print(f"[steer:easysteer] {model_key} {tag}: {cfg.rollouts} rollouts, layers {layers}")
    return _run_with_driver(model_key, cfg, _BatchDriver(flush, batch), tag)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def run_steering_grid(model_key: str, labels: list[str], strengths: list[float], layers: list[int], backend: str = "hf",
                      rollouts: int = 40, max_tokens: int = 2048, judge: bool = True, include_baseline: bool = True, batch: int = 8) -> list[Path]:
    cfg = SpiralConfig(rollouts=rollouts, extra_puzzle_rollouts=0, max_tokens=max_tokens)
    outs: list[Path] = []
    bundle = load_model(model_key) if backend == "hf" else None
    cells = ([(labels[0], 0.0)] if include_baseline else []) + [(lab, s) for lab in labels for s in strengths if s != 0]
    for label, s in cells:
        t0 = time.time()
        if backend == "hf":
            p = run_steered_hf(model_key, label, layers, s, cfg, batch=batch, model_bundle=bundle)
        else:
            p = run_steered_easysteer(model_key, label, layers, s, cfg)
        print(f"[steer] {p.parent.name}: done in {time.time() - t0:.0f}s")
        outs.append(p)
    if judge:
        from dprobe.judge import judge_transcripts, summarize

        summary = {}
        for p in outs:
            judge_transcripts(p, "frustration")
            s_ = summarize(p)
            summary[p.parent.name] = s_
            print(p.parent.name, "turn8:", s_.get(8), "all:", s_.get("all"))
        out_dir = RESULTS_DIR / "steer" / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"summary_{backend}.json", "w") as f:
            json.dump(summary, f, indent=1)
    return outs
