"""Two follow-ups to Finding 8 (Gemma 3 27B IT).

A. Which SAE features carry the assistant axis's spiral effect?
   Pushing Gemma 3 off its assistant axis (-1x at layers 34-46) makes it spiral, and coordinate 2733 plays no part.
   Decompose the layer-40 axis (2733 removed) into Gemma Scope 2 features (65k, medium L0) by orthogonal matching
   pursuit, then push along single features, each at exactly the norm of the axis push at every layer:
     axisfeat-control        the -1x axis push without coordinate 2733 (positive control)
     axisfeat-f<id>          one feature's decoder direction, signed the way the -axis push uses it
     axisfeat-random         a random direction of the same norm (negative control)
   16 conversations per cell, 1024 tokens per reply, as in every steering cell. Judge locally.

B. What is coordinate 2733 for?
   - where it appears: its value at every layer, at the first token vs the rest
   - next-token loss on held-out stories: baseline vs 2733 zeroed vs held at its per-layer mean (from emergence on)
   - attention sink: share of attention on the first token at a few layers, baseline vs zeroed (eager attention)

Output: RESULTS_DIR/axisfeat/gemma3_27b/{decomposition.json, name2733.json}; spiral cells under spiral/gemma3_27b/.
"""

from __future__ import annotations

import contextlib
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from dprobe.config import RESULTS_DIR, SpiralConfig, analysis_layers
from dprobe.models import capture, decoder_layers, load_model
from dprobe.stories import load_story_set

D = 2733
SAE_LAYER = 40


def out_dir(key):
    d = RESULTS_DIR / "axisfeat" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def omp(target: torch.Tensor, atoms: torch.Tensor, k: int):
    """Greedy orthogonal matching pursuit: target [H] ~ sum_j c_j atoms[idx_j] (atoms unit rows [F, H])."""
    idx, resid = [], target.clone()
    for _ in range(k):
        j = int((atoms @ resid).abs().argmax())
        if j in idx:
            break
        idx.append(j)
        A = atoms[idx].T                                            # [H, n]
        c = torch.linalg.lstsq(A, target.unsqueeze(1)).solution.squeeze(1)
        resid = target - A @ c
    return idx, c, float(resid.norm() ** 2 / target.norm() ** 2)


def decompose_and_steer(model_key="gemma3_27b", bundle=None, n_features=4, k_omp=30, rollouts=16, max_tokens=1024, batch=8, smoke=False):
    from dprobe.steer import run_steered_hf, steering_vectors

    model, tok, spec = bundle or load_model(model_key)
    device = next(model.parameters()).device
    sfx = "_smoke" if smoke else ""
    out = out_dir(model_key + sfx)
    layers = [l for l in analysis_layers(spec) if abs(l - spec.two_thirds_layer) <= 6]           # 34..46
    push = steering_vectors(model_key, "assistant_axis", layers, -1.0)                              # the push that spirals
    for l in layers:
        push[l] = push[l].clone(); push[l][D] = 0.0
    P = load_file(hf_hub_download("google/gemma-scope-2-27b-it", f"resid_post/layer_{SAE_LAYER}_width_65k_l0_medium/params.safetensors"))
    W = P["w_dec"].to(device)
    Wn = W / W.norm(dim=1, keepdim=True)
    Wn[:, D] = 0.0                                                    # compare directions with 2733 removed, like the push
    left = Wn.norm(dim=1, keepdim=True)
    Wn = torch.where(left >= 0.5, Wn / left.clamp_min(1e-6), torch.zeros_like(Wn))   # drop features that were mostly 2733
    t = push[SAE_LAYER].to(device).float()
    idx, coef, resid = omp(t, Wn, k_omp)
    order = sorted(range(len(idx)), key=lambda j: -abs(float(coef[j])))
    cos = (Wn @ (t / t.norm())).cpu()
    top_cos = cos.abs().topk(15).indices.tolist()
    dec = {"layer": SAE_LAYER, "steer_layers": layers, "omp_k": len(idx), "omp_residual_fraction": resid,
           "omp": [{"feature": idx[j], "coef": float(coef[j]), "cos_with_push": float(cos[idx[j]])} for j in order],
           "top_cos": [{"feature": i, "cos_with_push": float(cos[i])} for i in top_cos]}
    json.dump(dec, open(out / "decomposition.json", "w"), indent=1)
    print(f"[axisfeat] layer-{SAE_LAYER} push (no {D}) ~ {len(idx)} features, residual {resid:.2f}; top by |coef|: "
          + ", ".join(f"{idx[j]} ({float(coef[j]):+.2f}, cos {float(cos[idx[j]]):+.2f})" for j in order[:8]))
    cfg = SpiralConfig(rollouts=2 if smoke else rollouts, extra_puzzle_rollouts=0, max_tokens=128 if smoke else max_tokens)
    cells = {"axisfeat-control": push}
    for j in order[: (1 if smoke else n_features)]:
        f, s = idx[j], float(torch.sign(coef[j]))
        u = (s * Wn[f]).cpu()                                          # the direction the push moves along this feature
        cells[f"axisfeat-f{f}"] = {l: u * push[l].norm() for l in layers}
    g = torch.Generator().manual_seed(0)
    r = torch.randn(spec.hidden, generator=g); r[D] = 0.0; r = r / r.norm()
    cells["axisfeat-random"] = {l: r * push[l].norm() for l in layers}
    for tag, vecs in cells.items():
        run_steered_hf(model_key, tag, layers, 1.0, cfg=cfg, batch=batch, model_bundle=(model, tok, spec), vecs=vecs, tag=f"{tag}@34-46{sfx}")
    json.dump({**dec, "cells": list(cells)}, open(out / "decomposition.json", "w"), indent=1)


@contextlib.contextmanager
def set_coord(model, layers, dim, values):
    blocks = decoder_layers(model)
    hs = []

    def mk(v):
        def hook(_m, _i, o):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[..., dim] = v
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook
    for l in layers:
        hs.append(blocks[l].register_forward_hook(mk(values[l])))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def name_2733(model_key="gemma3_27b", n_loss=100, n_attn=10, smoke=False):
    sfx = "_smoke" if smoke else ""
    out = out_dir(model_key + sfx)
    model, tok, spec = load_model(model_key)
    device = next(model.parameters()).device
    L = list(range(spec.n_blocks))
    stories = [t for v in load_story_set(model_key, "emotions").values() for t in v[-3:]][: (4 if smoke else n_loss)]
    enc = [tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0] for t in stories]
    # 1. where does it appear?
    first, rest = torch.zeros(len(L)), torch.zeros(len(L))
    for ids in enc[:20]:
        with capture(model, L) as acts:
            model(input_ids=ids.unsqueeze(0).to(device))
        for l in L:
            first[l] += float(acts[l][0, 0, D]) / min(20, len(enc)); rest[l] += float(acts[l][0, 1:, D].float().mean()) / min(20, len(enc))
        del acts
    emerge = next(l for l in L if abs(rest[l]) > 100)
    # 2. next-token loss: baseline, zeroed, mean-held (from emergence on)
    def loss(ctx):
        tot, n = 0.0, 0
        for ids in enc:
            x = ids.unsqueeze(0).to(device)
            with ctx():
                logits = model(input_ids=x).logits[0, :-1].float()
            tot += float(torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum")); n += x.shape[1] - 1
        return tot / n
    later = [l for l in L if l >= emerge]
    res = {"baseline": loss(contextlib.nullcontext),
           "zeroed": loss(lambda: set_coord(model, later, D, {l: 0.0 for l in later})),
           "mean_held": loss(lambda: set_coord(model, later, D, {l: float(rest[l]) for l in later}))}
    # control: zero an ordinary coordinate of typical size
    ctrl = 100
    res["zeroed_ordinary_coord_100"] = loss(lambda: set_coord(model, later, ctrl, {l: 0.0 for l in later}))
    print(f"[name2733] emerges at layer {emerge}; next-token loss {res}")
    del model
    import gc
    gc.collect(); torch.cuda.empty_cache()
    # 3. attention sink (eager attention returns weights)
    model, tok, spec = load_model(model_key, attn="eager")
    probe_layers = [5, 10, 20, 30, 40, 50]
    sink = {"baseline": {}, "zeroed": {}}
    for name, ctx in [("baseline", contextlib.nullcontext), ("zeroed", lambda: set_coord(model, later, D, {l: 0.0 for l in later}))]:
        acc = {l: 0.0 for l in probe_layers}
        for ids in enc[: (2 if smoke else n_attn)]:
            with ctx():
                o = model(input_ids=ids[:256].unsqueeze(0).to(device), output_attentions=True)
            for l in probe_layers:
                a = o.attentions[l][0].float()                         # [heads, q, k]
                acc[l] += float(a[:, 1:, 0].mean()) / (2 if smoke else n_attn)
        sink[name] = acc
    print(f"[name2733] share of attention on the first token: {sink}")
    json.dump({"emergence_layer": emerge, "value_first_token": first.tolist(), "value_other_tokens": rest.tolist(),
               "loss_nats_per_token": res, "attention_on_first_token": sink}, open(out / "name2733.json", "w"), indent=1)
