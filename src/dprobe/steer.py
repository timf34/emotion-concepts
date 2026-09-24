"""Run the extended elicitation under activation steering and judge the result.

Two backends:
  hf        : forward hooks add a scaled unit vector to the residual stream at the chosen layers during
              model.generate (reference implementation; batches conversations per turn; slow-ish).
  easysteer : ZJU-REAL/EasySteer (vLLM 0.29 fork). Vectors come from the gguf files written by
              extract.export_gguf; steering is passed per request via SteeringSpec. Much faster.

Steering strength (mode "vec", default): multiples of each layer's own difference-of-means vector norm, so
+1 adds one "story-difference" along unit(v_l). Mode "resid" (Anthropic's convention, fraction of residual
norm) is kept but is WRONG for Gemma: its residual norm (~60,000 at L40) is two massive-activation dimensions,
so 0.06 of it is 5-7x the vector and produces gibberish (issue 28). Calibrate the multiplier with
`dprobe.cli coherence` on a short sweep before running a grid.

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


def steering_vectors(model_key: str, label: str, layers: list[int], strength: float, mode: str = "vec") -> dict[int, torch.Tensor]:
    """mode "vec":   {layer: strength * v_l}                    (multiples of the difference-of-means vector)
       mode "resid": {layer: strength * ||resid_l|| * unit(v_l)} (fraction of residual norm; unsuitable for Gemma)"""
    vecs = load_vectors(model_key, _set_for(label), denoised=True)[label]
    missing = [l for l in layers if l not in vecs]
    if missing:
        raise ValueError(f"no denoised vector at layers {missing} (available: {sorted(vecs)})")
    if mode == "vec":
        return {l: strength * vecs[l] for l in layers}
    norms = residual_norms(model_key)
    return {l: strength * norms[l] * _unit(vecs[l]) for l in layers}


def tag_for(label: str, layers: list[int], strength: float, mode: str = "vec") -> str:
    ls = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])
    sep = "v" if mode == "vec" else "x"
    return f"steer-{label.replace(' ', '_')}@{ls}{sep}{strength:+g}"


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
                   positions: str = "all", batch: int = 8, model_bundle=None, mode: str = "vec") -> Path:
    cfg = cfg or SpiralConfig()
    model, tok, spec = model_bundle or load_model(model_key)
    vecs = steering_vectors(model_key, label, layers, strength, mode) if strength != 0 else {}
    device = next(model.parameters()).device
    tag = tag_for(label, layers, strength, mode)

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
                          gpu_mem: float = 0.9, max_model_len: int = 20000, batch: int = 64, mode: str = "vec") -> Path:
    from vllm import SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

    cfg = cfg or SpiralConfig()
    spec = get_model(model_key)
    tag = tag_for(label, layers, strength, mode)
    llm = _easysteer_llm(spec.hf_id, gpu_mem, max_model_len)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens)

    steering = False
    if strength != 0:
        gguf_path = vectors_dir(model_key) / "gguf" / _set_for(label) / f"{label.replace(' ', '_')}.gguf"
        if not gguf_path.exists():
            raise FileNotFoundError(gguf_path)
        # gguf holds the raw denoised difference-of-means; EasySteer applies one scalar to it.
        if mode == "vec":
            scale = float(strength)
        else:
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
                      rollouts: int = 40, max_tokens: int = 2048, judge: bool = True, include_baseline: bool = True, batch: int = 8,
                      mode: str = "vec", model_bundle=None) -> list[Path]:
    cfg = SpiralConfig(rollouts=rollouts, extra_puzzle_rollouts=0, max_tokens=max_tokens)
    outs: list[Path] = []
    bundle = model_bundle or (load_model(model_key) if backend == "hf" else None)
    cells = ([(labels[0], 0.0)] if include_baseline else []) + [(lab, s) for lab in labels for s in strengths if s != 0]
    for label, s in cells:
        t0 = time.time()
        if backend == "hf":
            p = run_steered_hf(model_key, label, layers, s, cfg, batch=batch, model_bundle=bundle, mode=mode)
        else:
            p = run_steered_easysteer(model_key, label, layers, s, cfg, mode=mode)
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
        sp = out_dir / f"summary_{backend}.json"
        if sp.exists():
            try:
                summary = {**json.load(open(sp)), **summary}
            except Exception:  # noqa: BLE001
                pass
        with open(sp, "w") as f:
            json.dump(summary, f, indent=1)
    return outs


# ---------------------------------------------------------------------------
# Coherence check (for calibrating the multiplier)
# ---------------------------------------------------------------------------
def coherence(transcripts_path: Path) -> dict:
    """Cheap degeneration metrics over assistant turns: distinct-word ratio and share of repeated 3-grams.
    Coherent Gemma text: distinct ratio ~0.5-0.7, repeated-3gram share < 0.2. Gibberish: ratio < 0.2, share > 0.6."""
    import re

    from dprobe.spiral import load_transcripts

    convs = load_transcripts(transcripts_path)
    ratios, rep = [], []
    for c in convs:
        for m in c["messages"]:
            if m["role"] != "assistant":
                continue
            w = re.findall(r"[A-Za-z']+", m["content"].lower())
            if len(w) < 20:
                continue
            ratios.append(len(set(w)) / len(w))
            tg = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
            from collections import Counter
            cnt = Counter(tg)
            rep.append(sum(v for v in cnt.values() if v > 1) / max(1, len(tg)))
    import numpy as np
    return {"n_turns": len(ratios), "distinct_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "repeated_3gram_share": float(np.mean(rep)) if rep else float("nan")}


def calibrate(model_key: str, label: str, layers: list[int], multipliers: list[float], backend: str = "hf", rollouts: int = 2,
              max_tokens: int = 1024, batch: int = 8, rel_ratio: float = 0.8, rel_rep: float = 0.15, model_bundle=None) -> float:
    """Largest multiplier that stays coherent *relative to the model's own unsteered baseline* (a model whose
    baseline is repetitive, like Gemma 4 grinding through arithmetic, must not be judged by absolute thresholds).
    Coherent := distinct_ratio >= rel_ratio * baseline_ratio and repeated_3gram <= baseline_rep + rel_rep.
    Runs at the grid's generation length so long-context degeneration is caught."""
    cfg = SpiralConfig(rollouts=rollouts, extra_puzzle_rollouts=0, max_tokens=max_tokens)
    bundle = model_bundle or (load_model(model_key) if backend == "hf" else None)

    def run(m):
        if backend == "hf":
            return run_steered_hf(model_key, label, layers, m, cfg, batch=batch, model_bundle=bundle, mode="vec")
        return run_steered_easysteer(model_key, label, layers, m, cfg, mode="vec")

    base = coherence(run(0.0))
    print(f"[calibrate] {label} baseline: distinct_ratio={base['distinct_ratio']:.2f} repeated_3gram={base['repeated_3gram_share']:.2f}")
    ok, results = [0.0], {"0": base}
    for m in sorted(multipliers):
        c = coherence(run(m))
        results[str(m)] = c
        good = c["distinct_ratio"] >= rel_ratio * base["distinct_ratio"] and c["repeated_3gram_share"] <= base["repeated_3gram_share"] + rel_rep
        print(f"[calibrate] {label} x{m:+g}: distinct_ratio={c['distinct_ratio']:.2f} repeated_3gram={c['repeated_3gram_share']:.2f} -> {'ok' if good else 'DEGENERATE'}")
        if good:
            ok.append(m)
        else:
            break                     # stronger multipliers will not recover
    best = max(ok)
    out_dir = RESULTS_DIR / "steer" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"calibration_{label}.json", "w") as f:
        json.dump({"layers": layers, "max_tokens": max_tokens, "results": results, "chosen": best}, f, indent=1)
    print(f"[calibrate] {label}: chosen multiplier {best}")
    return best
