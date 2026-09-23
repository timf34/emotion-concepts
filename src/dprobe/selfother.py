"""Self vs other dissociation on the Pain-axis 420-scenario set (Tagliabue et al. 2026, MIT).

Scenarios are short multi-turn "[User]: ... [Assistant]:" transcripts in 21 categories:
11 harm-directed-at-the-model (incl. repeated rejection of its work), 5 user-suffering
(incl. psychological crisis), 5 controls. We render them with the chat template, read every
vector at the final (response-preparation) token and as the mean over user tokens, and
z-score within model across the pool. Output: RESULTS_DIR/selfother/<model_key>/selfother.pt
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from dprobe.config import DATA_DIR, RESULTS_DIR, ExtractConfig, analysis_layers
from dprobe.models import capture, load_model
from dprobe.probe import assemble_vector_bank
from dprobe.transcripts import render

_TURN_RE = re.compile(r"\[(User|Assistant)\]:\s*", re.I)


def to_messages(text: str) -> list[dict]:
    parts = _TURN_RE.split(text)
    msgs = []
    for i in range(1, len(parts), 2):
        role = "user" if parts[i].lower() == "user" else "assistant"
        content = parts[i + 1].strip()
        if content:
            msgs.append({"role": role, "content": content})
    if not msgs or msgs[-1]["role"] != "user":
        # scenario ends on an empty "[Assistant]:" -> we probe at the generation prompt
        pass
    return msgs


@torch.no_grad()
def run_selfother(model_key: str, layers: list[int] | None = None, model_bundle=None, limit: int | None = None, vectors_from: str | None = None) -> Path:
    model, tok, spec = model_bundle or load_model(model_key)
    layers = layers or analysis_layers(spec, ExtractConfig())
    labels, bank = assemble_vector_bank(vectors_from or model_key, layers)
    device = next(model.parameters()).device
    bank = {l: v.to(device=device, dtype=torch.float32) for l, v in bank.items()}
    with open(DATA_DIR / "pain_axis" / "self_other_420_scenarios.json") as f:
        scenarios = json.load(f)
    if limit:
        # keep category balance: round-robin over categories
        by_cat: dict[str, list] = {}
        for s_ in scenarios:
            by_cat.setdefault(s_["category"], []).append(s_)
        picked, i = [], 0
        while len(picked) < limit and any(by_cat.values()):
            for k in list(by_cat):
                if by_cat[k] and len(picked) < limit:
                    picked.append(by_cat[k].pop(0))
        scenarios = picked
    rows = []
    for s in tqdm(scenarios, desc="self/other"):
        msgs = to_messages(s["text"])
        if not msgs:
            continue
        r = render(tok, msgs, add_generation_prompt=True)
        ids = r.input_ids.unsqueeze(0).to(device)
        with capture(model, layers) as acts:
            model(input_ids=ids)
        user_idx = [i for t in r.turns if t["role"] == "user" for i in range(t["start"], t["end"])]
        asst_idx = [i for t in r.turns if t["role"] == "assistant" for i in range(t["start"], t["end"])]
        rec = {k: s[k] for k in ("id", "category", "stratum", "perspective", "intensity")}
        rec["final"], rec["user_mean"], rec["asst_mean"] = {}, {}, {}
        for l in layers:
            P = acts[l][0].float() @ bank[l].T
            rec["final"][l] = P[-1].cpu()
            rec["user_mean"][l] = P[user_idx].mean(0).cpu() if user_idx else None
            rec["asst_mean"][l] = P[asst_idx].mean(0).cpu() if asst_idx else None
        rows.append(rec)
    out = RESULTS_DIR / "selfother" / (vectors_from or model_key)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": labels, "layers": layers, "rows": rows}, out / "selfother.pt")
    print(f"[selfother] {len(rows)} scenarios -> {out / 'selfother.pt'}")
    return out / "selfother.pt"
