"""Project spiral transcripts (and other conversations) onto emotion / syndrome / pain vectors.

For every conversation and every analysis layer we store, per assistant turn:
  proj_mean  : mean over the assistant's content tokens of dot(h, v)          [E]
  proj_prep  : dot(h, v) at the response-preparation token                    [E]
  proj_user  : mean over the preceding user turn's tokens                     [E]
  act_mean   : mean assistant-token activation itself (bf16)                  [H]   (for the on-policy spiral direction)
  act_prep   : activation at the response-preparation token (bf16)            [H]
Plus per-token projections at ONE layer for the first `token_level_convs` conversations (for plots).

Output: RESULTS_DIR/probe/<model_key>/<condition>[_tag]/probe.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from dprobe.config import RESULTS_DIR, ExtractConfig, analysis_layers, get_model
from dprobe.extract import load_pain_axis, load_vectors, vectors_dir
from dprobe.models import capture, load_model
from dprobe.spiral import load_transcripts
from dprobe.transcripts import render


def assemble_vector_bank(model_key: str, layers: list[int], denoised: bool = True) -> tuple[list[str], dict[int, torch.Tensor]]:
    """All vectors we project onto, stacked per layer: emotions + syndromes (+ pain axis when available)."""
    labels: list[str] = []
    bank: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for set_name in ("emotions", "syndromes"):
        try:
            vecs = load_vectors(model_key, set_name, denoised)
        except FileNotFoundError:
            print(f"[probe] no {set_name} vectors for {model_key}")
            continue
        for e, by_l in vecs.items():
            if all(l in by_l for l in layers):
                labels.append(e)
                for l in layers:
                    bank[l].append(by_l[l])
    pain = load_pain_axis(model_key)
    if pain is not None:
        pv = _pain_axis_by_layer(pain)
        if pv and all(l in pv for l in layers):
            labels.append("pain_axis")
            for l in layers:
                bank[l].append(pv[l])
    return labels, {l: torch.stack(v) for l, v in bank.items()}


def _pain_axis_by_layer(obj) -> dict[int, torch.Tensor] | None:
    """Best-effort reader for the Pain-axis repo's pain_vectors.pt (structure inspected on the pod)."""
    if isinstance(obj, dict):
        # common layouts: {"S2_1P": {"mean": {layer: vec}}} or {layer: vec}
        for k in ("S2_1P", "S1_1P"):
            if k in obj:
                sub = obj[k]
                if isinstance(sub, dict):
                    if "mean" in sub and isinstance(sub["mean"], dict):
                        return {int(l): torch.as_tensor(v).float() for l, v in sub["mean"].items() if _is_vec(v)}
                    if all(_is_vec(v) for v in sub.values()):
                        return {int(l): torch.as_tensor(v).float() for l, v in sub.items()}
        if all(isinstance(k, int) or str(k).isdigit() for k in obj) and all(_is_vec(v) for v in obj.values()):
            return {int(l): torch.as_tensor(v).float() for l, v in obj.items()}
    return None


def _is_vec(v) -> bool:
    return hasattr(v, "shape") and len(v.shape) == 1


def probe_dir(model_key: str, condition: str, tag: str = "") -> Path:
    d = RESULTS_DIR / "probe" / model_key / (condition + (f"_{tag}" if tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


@torch.no_grad()
def probe_transcripts(
    model_key: str,
    transcripts_path: Path,
    condition: str = "extended",
    tag: str = "",
    layers: list[int] | None = None,
    token_level_convs: int = 12,
    token_level_layer: int | None = None,
    max_tokens: int = 16384,
    denoised: bool = True,
    model_bundle=None,
) -> Path:
    model, tok, spec = model_bundle or load_model(model_key)
    layers = layers or analysis_layers(spec, ExtractConfig())
    labels, bank = assemble_vector_bank(model_key, layers, denoised)
    device = next(model.parameters()).device
    bank = {l: v.to(device=device, dtype=torch.float32) for l, v in bank.items()}
    tl_layer = token_level_layer or spec.two_thirds_layer
    if tl_layer not in layers:
        layers = sorted(layers + [tl_layer])
    convs = load_transcripts(transcripts_path)
    print(f"[probe] {model_key}: {len(convs)} conversations, {len(labels)} vectors, {len(layers)} layers")

    E, H = len(labels), spec.hidden
    max_turns = max(sum(m["role"] == "assistant" for m in c["messages"]) for c in convs)
    N, L = len(convs), len(layers)
    proj_mean = torch.full((N, max_turns, L, E), float("nan"))
    proj_prep = torch.full((N, max_turns, L, E), float("nan"))
    proj_user = torch.full((N, max_turns, L, E), float("nan"))
    act_mean = torch.zeros((N, max_turns, L, H), dtype=torch.bfloat16)
    act_prep = torch.zeros((N, max_turns, L, H), dtype=torch.bfloat16)
    n_turns = torch.zeros(N, dtype=torch.long)
    token_level = []

    for ci, c in enumerate(tqdm(convs, desc="probe")):
        r = render(tok, c["messages"])
        ids = r.input_ids[:max_tokens].unsqueeze(0).to(device)
        with capture(model, layers) as acts:
            model(input_ids=ids)
        asst = [t for t in r.turns if t["role"] == "assistant" and t["end"] <= ids.shape[1]]
        users = [t for t in r.turns if t["role"] == "user"]
        n_turns[ci] = len(asst)
        for li, l in enumerate(layers):
            h = acts[l][0].float()                       # [T, H]
            P = h @ bank[l].T                            # [T, E]
            for t in asst:
                k = t["turn_idx"]
                proj_mean[ci, k, li] = P[t["start"]:t["end"]].mean(0).cpu()
                proj_prep[ci, k, li] = P[t["prep"]].cpu()
                act_mean[ci, k, li] = h[t["start"]:t["end"]].mean(0).to(torch.bfloat16).cpu()
                act_prep[ci, k, li] = h[t["prep"]].to(torch.bfloat16).cpu()
                if k < len(users):
                    u = users[k]
                    proj_user[ci, k, li] = P[u["start"]:u["end"]].mean(0).cpu()
            if l == tl_layer and ci < token_level_convs:
                token_level.append({"id": c["id"], "layer": l, "tokens": tok.convert_ids_to_tokens(ids[0].tolist()), "proj": P.cpu().half(), "turns": r.turns})
        del acts

    out = probe_dir(model_key, condition, tag)
    torch.save(
        {
            "model": model_key, "labels": labels, "layers": layers, "ids": [c["id"] for c in convs], "puzzles": [c["puzzle"] for c in convs],
            "n_turns": n_turns, "proj_mean": proj_mean, "proj_prep": proj_prep, "proj_user": proj_user,
            "act_mean": act_mean, "act_prep": act_prep, "token_level": token_level, "denoised": denoised,
        },
        out / "probe.pt",
    )
    print(f"[probe] saved -> {out / 'probe.pt'}")
    return out / "probe.pt"


@torch.no_grad()
def quantity_sweep(model_key: str, layers: list[int] | None = None, model_bundle=None) -> Path:
    """Anthropic's 'emotion tracks a swept quantity' validation (Tylenol dose, days dog missing, ...)."""
    from dprobe.config import DATA_DIR

    model, tok, spec = model_bundle or load_model(model_key)
    layers = layers or analysis_layers(spec, ExtractConfig())
    labels, bank = assemble_vector_bank(model_key, layers)
    device = next(model.parameters()).device
    bank = {l: v.to(device=device, dtype=torch.float32) for l, v in bank.items()}
    with open(DATA_DIR / "prompt_templates_vary_quantity.json") as f:
        templates = json.load(f)["templates"]
    rows = []
    for name, t in templates.items():
        for x in t["values"]:
            msgs = [{"role": "user", "content": t["prompt"].format(X=x)}]
            r = render(tok, msgs, add_generation_prompt=True)
            ids = r.input_ids.unsqueeze(0).to(device)
            with capture(model, layers) as acts:
                model(input_ids=ids)
            for li, l in enumerate(layers):
                P = acts[l][0].float() @ bank[l].T
                rows.append({"template": name, "x": x, "layer": l, "prep": P[-1].cpu(), "user_mean": P[1:-1].mean(0).cpu()})
    out = vectors_dir(model_key) / "quantity_sweep.pt"
    torch.save({"labels": labels, "rows": rows}, out)
    print(f"[probe] quantity sweep -> {out}")
    return out
