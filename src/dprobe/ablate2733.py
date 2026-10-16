"""Does removing Gemma 3's massive coordinate 2733 inside the model change (a) the persona-emotion cosine and (b) the spiral?

The coordinate builds up across layers (about 400 at layer 0, 15,000 at layer 24, 87,000 at layer 61), so it is HELD at a
fixed multiple of its own per-layer average at every block output (all positions), rather than scaled layer by layer
(which would compound):
    hold x1.0   its variation removed, its size kept
    hold x0.5   half its usual size
    hold x0.0   removed
(a) vectors: emotion vectors + re-encoded assistant axis recorded with hold x0.0, saved as model key gemma3_27b_no2733.
(b) cells: the rejection eval under each hold, plus a fresh unsteered control, 32 conversations each (Countdown-156,
    1024 tokens per reply, batch 8), tags hold2733x<s>@all and ctrl32-unsteered. Judge locally.
"""

from __future__ import annotations

import contextlib
import json

import torch

from dprobe.config import RESULTS_DIR, SpiralConfig
from dprobe.models import capture, decoder_layers, load_model
from dprobe.stories import load_story_set

D = 2733


@torch.no_grad()
def layer_means(bundle, n=20):
    model, tok, spec = bundle
    device = next(model.parameters()).device
    L = list(range(spec.n_blocks))
    texts = [t for v in load_story_set("gemma3_27b", "emotions").values() for t in v[-1:]][:n]
    m = torch.zeros(len(L), dtype=torch.float64)
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        with capture(model, L) as acts:
            model(input_ids=ids)
        for l in L:
            m[l] += float(acts[l][0, 1:, D].float().mean()) / len(texts)
        del acts
    return {l: float(m[l]) for l in L}


@contextlib.contextmanager
def hold(model, means: dict[int, float], scale: float):
    blocks = decoder_layers(model)
    hs = []

    def mk(v):
        def hook(_m, _i, o):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[..., D] = v
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook
    for l, b in enumerate(blocks):
        hs.append(b.register_forward_hook(mk(scale * means[l])))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def run(stages=("vectors", "cells"), cells=("ctrl", "1.0", "0.5", "0.0"), rollouts=32, smoke=False, sync=True):
    from dprobe.axis_reencode import import_reencoded_axis, reencode_axis  # noqa: F401
    from dprobe.extract import run_extract
    from dprobe.steer import run_steered_hf
    from dprobe.sync import sync_up

    bundle = load_model("gemma3_27b")
    model = bundle[0]
    means = layer_means(bundle, n=4 if smoke else 20)
    sfx = "_smoke" if smoke else ""
    out = RESULTS_DIR / "ablate2733"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"layer_means": means}, open(out / f"means{sfx}.json", "w"), indent=1)
    print(f"[ablate2733] per-layer means of coordinate {D}: L0 {means[0]:.0f}, L16 {means[16]:.0f}, L24 {means[24]:.0f}, L61 {means[61]:.0f}")
    if "vectors" in stages:
        key = "gemma3_27b_no2733" + sfx
        with hold(model, means, 0.0):
            run_extract("gemma3_27b", ("emotions",), model_bundle=bundle, smoke=smoke, out_key=key)
            reencode_axis("gemma3_27b", bundle, k_role=2 if smoke else 50, n_default=4 if smoke else 300, roles_limit=6 if smoke else None, out_key=key)
        if sync:
            sync_up(("vectors",), models=(key,))
    if "cells" in stages:
        cfg = SpiralConfig(rollouts=2 if smoke else rollouts, extra_puzzle_rollouts=0, max_tokens=128 if smoke else 1024)
        for c in cells:
            if c == "ctrl":
                run_steered_hf("gemma3_27b", "ctrl", [], 0, cfg=cfg, batch=8, model_bundle=bundle, vecs={}, tag=f"ctrl32-unsteered{sfx}")
            else:
                with hold(model, means, float(c)):
                    run_steered_hf("gemma3_27b", f"hold{c}", [], 0, cfg=cfg, batch=8, model_bundle=bundle, vecs={}, tag=f"hold2733x{c}@all{sfx}")
            if sync:
                sync_up(("spiral",), models=("gemma3_27b",))
    if sync:
        sync_up(("ablate2733",))
