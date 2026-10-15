"""What is Gemma 3's massive coordinate 2733, and does the spiral run through it?

Gemma 3 27B IT's assistant-axis/emotion "entanglement" is carried almost entirely by one massive-activation coordinate
(2733; ~15,000 at layer 24 where typical coordinates are ~2.5). This job collects, in one model load:

  1. SAE features (Gemma Scope 2, resid_post, layers 16 and 31) averaged per text, for
       - the held-out emotion stories (last 60 per emotion, the ones never used to build the vectors),
       - a role-play subsample (axis_reencode selection: 10 score-3 responses per role + 100 default),
       - the assistant turns of 40 of Gemma 3's own spiral transcripts,
     so the assistant/persona contrast and the calm/agitated contrast can be compared feature by feature.
  2. Per-token values of the massive coordinates (2733, 1687, 104) at layers 16, 20, 24, 31, with token ids,
     to see on which tokens they move.
  3. Causal cells in the rejection eval (16 conversations each, 1024 tokens, like every steering cell):
       axis_no2733   the -2x assistant-axis push at layers 20-26 with coordinate 2733 zeroed
       axis_only2733 only the axis's coordinate-2733 component of that push
       clamp2733     no steering; coordinate 2733 clamped at the default assistant's mean at layers 16-26
     (the full -2x axis push at 20-26 already exists: extended_steer-assistant_axis@20-26v-2, judge 5.8 vs 4.2).

Output: RESULTS_DIR/sae2733/gemma3_27b/{stories,roles,spiral}_L{layer}.pt, tokens.pt, meta.json; spiral cells under
spiral/gemma3_27b/extended_<tag>. Judge the cells locally afterwards.
"""

from __future__ import annotations

import contextlib
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

from dprobe.config import RESULTS_DIR, SpiralConfig
from dprobe.models import capture, decoder_layers, load_model
from dprobe.spiral import load_transcripts, transcripts_path
from dprobe.stories import load_story_set
from dprobe.transcripts import render

SAE_REPO = "google/gemma-scope-2-27b-it"
COORDS = (2733, 1687, 104)
COORD_LAYERS = (16, 20, 24, 31)


def load_sae(layer: int, width: str, l0: str, device):
    p = load_file(hf_hub_download(SAE_REPO, f"resid_post/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"))
    return {k: v.to(device) for k, v in p.items()}


def encode(sae, h):                                   # h [T, H] float32 -> [T, width]
    pre = h @ sae["w_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


@contextlib.contextmanager
def clamp(model, layers, dim: int, targets: dict[int, float]):
    """Set residual coordinate `dim` to targets[layer] at the output of each block in `layers`."""
    blocks = decoder_layers(model)
    handles = []

    def mk(t):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[..., dim] = t
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    for l in layers:
        handles.append(blocks[l].register_forward_hook(mk(targets[l])))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def collect(model_key="gemma3_27b", bundle=None, sae_layers=(16, 31), width="65k", l0="medium", n_story=60, n_role=10,
            n_default=100, n_spiral=40, smoke=False):
    from dprobe.axis_reencode import select_conversations

    model, tok, spec = bundle or load_model(model_key)
    device = next(model.parameters()).device
    out = RESULTS_DIR / "sae2733" / (model_key + ("_smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    saes = {L: load_sae(L, width, l0, device) for L in sae_layers}
    layers = sorted(set(sae_layers) | set(COORD_LAYERS))

    stories = load_story_set(model_key, "emotions")
    items = []                                        # (set, label, ids [T], spans [(start, end, turn)])
    for e, texts in stories.items():
        for t in texts[-(2 if smoke else n_story):]:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=1024)["input_ids"][0]
            if len(ids) > 60:
                items.append(("stories", e, ids, [(50, len(ids), 0)]))      # Anthropic: average from the 50th token
    convs = select_conversations(1 if smoke else n_role, 4 if smoke else n_default, roles_limit=6 if smoke else None)
    for role, cs in convs.items():
        for c in cs:
            r = render(tok, c)
            a = [t for t in r.turns if t["role"] == "assistant"][-1]
            ids = r.input_ids[:2048]
            if min(a["end"], len(ids)) - a["start"] > 1:
                items.append(("roles", role, ids, [(a["start"], min(a["end"], len(ids)), 0)]))
    for c in load_transcripts(transcripts_path(model_key, "extended"))[: (2 if smoke else n_spiral)]:
        r = render(tok, c["messages"])
        ids = r.input_ids[:16384]
        spans = [(t["start"], min(t["end"], len(ids)), t["turn_idx"]) for t in r.turns if t["role"] == "assistant" and t["start"] < len(ids)]
        items.append(("spiral", c["id"], ids, spans))

    feats = {(s, L): [] for s in ("stories", "roles", "spiral") for L in sae_layers}
    rows = {s: [] for s in ("stories", "roles", "spiral")}
    coord_tokens = []
    print(f"[sae2733] {len(items)} texts: " + ", ".join(f"{s} {sum(i[0] == s for i in items)}" for s in rows))
    for kind, label, ids, spans in tqdm(items, desc="sae2733"):
        with capture(model, layers) as acts:
            model(input_ids=ids.unsqueeze(0).to(device))
        for (s, e, turn) in spans:
            rows[kind].append({"label": label, "turn": turn, "n_tokens": e - s})
            for L in sae_layers:
                h = acts[L][0, s:e].float()
                feats[(kind, L)].append(encode(saes[L], h).mean(0).half().cpu())
        if kind != "spiral":
            s, e, _ = spans[0]
            coord_tokens.append({"set": kind, "label": label, "ids": ids[s:e].tolist(),
                                 "coords": {L: acts[L][0, s:e][:, list(COORDS)].float().cpu().half() for L in COORD_LAYERS}})
        else:
            coord_tokens.append({"set": kind, "label": label, "turn_means": {L: torch.stack([acts[L][0, s:e][:, list(COORDS)].float().mean(0) for s, e, _ in spans]).cpu() for L in COORD_LAYERS}})
        del acts
    for (kind, L), v in feats.items():
        if v:
            torch.save({"rows": rows[kind], "features": torch.stack(v)}, out / f"{kind}_L{L}.pt")
    torch.save({"coords": list(COORDS), "items": coord_tokens}, out / "tokens.pt")
    json.dump({"sae": f"{SAE_REPO} resid_post width {width} l0 {l0}", "sae_layers": list(sae_layers), "coord_layers": list(COORD_LAYERS),
               "n": {k: len(v) for k, v in rows.items()}}, open(out / "meta.json", "w"), indent=1)
    print(f"[sae2733] saved -> {out}")
    return out


def causal_cells(model_key="gemma3_27b", bundle=None, rollouts=16, max_tokens=1024, batch=8, smoke=False):
    from dprobe.steer import run_steered_hf, steering_vectors

    model, tok, spec = bundle or load_model(model_key)
    cfg = SpiralConfig(rollouts=2 if smoke else rollouts, extra_puzzle_rollouts=0, max_tokens=128 if smoke else max_tokens)
    layers = [20, 22, 24, 26]
    D = COORDS[0]
    full = steering_vectors(model_key, "assistant_axis", layers, -2.0)
    no = {l: v.clone() for l, v in full.items()}
    only = {l: torch.zeros_like(v) for l, v in full.items()}
    for l in layers:
        no[l][D] = 0.0
        only[l][D] = full[l][D]
    share = {l: float(full[l][D] ** 2 / (full[l] ** 2).sum()) for l in layers}
    print(f"[sae2733] share of the -2x axis push on coordinate {D} by layer: {share}")
    sfx = "_smoke" if smoke else ""
    run_steered_hf(model_key, "axis_no2733", layers, 1.0, cfg=cfg, batch=batch, model_bundle=bundle, vecs=no, tag=f"axis-no2733@20-26v-2{sfx}")
    run_steered_hf(model_key, "axis_only2733", layers, 1.0, cfg=cfg, batch=batch, model_bundle=bundle, vecs=only, tag=f"axis-only2733@20-26v-2{sfx}")
    # clamp at the default assistant's own mean on coordinate D (Gemma-Assistantness default vector, per layer)
    dv = torch.load(hf_hub_download("timf34/gemma-assistant-axis-vectors", "vectors/gemma-3-27b/default_vector.pt", repo_type="dataset")).float()
    cl = list(range(16, 27))
    targets = {l: float(dv[l][D]) for l in cl}
    print(f"[sae2733] clamp targets for coordinate {D}: {targets}")
    with clamp(model, cl, D, targets):
        run_steered_hf(model_key, "clamp2733", cl, 0.0, cfg=cfg, batch=batch, model_bundle=bundle, vecs={}, tag=f"clamp2733@16-26{sfx}")
    out = RESULTS_DIR / "sae2733" / (model_key + sfx)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"share_of_push_on_2733": share, "clamp_targets": targets}, open(out / "causal_meta.json", "w"), indent=1)
