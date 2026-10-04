"""Assistant Axis for any Gemma 3 27B variant, by re-reading Gemma 3's archived role-play responses.

The original axis (Lu et al. 2026 protocol, timf34/GemmaAssistantAxis) was built from Gemma 3 27B IT's own
responses under 275 role system prompts, judged 0-3 for how fully each response plays the role:
    role vector    = mean response activation over that role's score-3 responses
    default vector = mean over the default (no-persona) responses, unfiltered
    axis           = default - mean(role vectors)          (+ = more assistant-like)

For fine-tuned organisms (DPO / SFT) we do not regenerate: each model reads the SAME archived texts
(teacher-forced), so any change in the geometry comes from the weights alone. A fixed random subsample
(same seed for every model) keeps it cheap; plain Gemma 3 is re-encoded the same way as the control.

Activation of a response = mean over the assistant turn's content tokens of the residual stream after
each decoder block (index = block, as everywhere in dprobe).

Output: RESULTS_DIR/vectors/<model>/axis_reencoded/{assistant_axis.pt, default_vector.pt, role_vectors.pt, meta.json}
    role_vectors.pt = {"roles": [...], "vectors": [R, L, H] float32}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from dprobe.extract import load_vectors, vectors_dir
from dprobe.models import capture, load_model
from dprobe.transcripts import render

AXIS_RESULTS_REPO = "timf34/gemma-assistant-axis-results"
SOURCE = "gemma-3-27b"
SEED = 0


def _archive() -> Path:
    root = snapshot_download(AXIS_RESULTS_REPO, repo_type="dataset",
                             allow_patterns=[f"{SOURCE}/responses/*.jsonl", f"{SOURCE}/scores/*.json"])
    return Path(root) / SOURCE


def select_conversations(k_role: int, n_default: int, roles_limit: int | None = None) -> dict[str, list[list[dict]]]:
    """{role: [conversation, ...]}: k_role score-3 responses per role, n_default for 'default'. Deterministic."""
    base = _archive()
    rng = random.Random(SEED)
    out: dict[str, list[list[dict]]] = {}
    files = sorted((base / "responses").glob("*.jsonl"))
    roles = [p.stem for p in files if p.stem != "default"]
    if roles_limit:
        roles = roles[:roles_limit]
    for role in ["default"] + roles:
        rows = [json.loads(l) for l in open(base / "responses" / f"{role}.jsonl")]
        key = {f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}": r for r in rows}
        if role == "default":
            keys = sorted(key)
            n = n_default
        else:
            scores = json.load(open(base / "scores" / f"{role}.json"))
            keys = sorted(k for k, s in scores.items() if s == 3 and k in key)
            n = k_role
        keys = rng.sample(keys, min(n, len(keys)))
        out[role] = [key[k]["conversation"] for k in keys]
    return out


@torch.no_grad()
def reencode_axis(model_key: str, model_bundle=None, k_role: int = 50, n_default: int = 300, batch_size: int = 16,
                  max_length: int = 2048, roles_limit: int | None = None, out_key: str | None = None) -> Path:
    model, tok, spec = model_bundle or load_model(model_key)
    device = next(model.parameters()).device
    layers = list(range(spec.n_blocks))
    convs = select_conversations(k_role, n_default, roles_limit)
    items = []                                              # (role, ids [T], start, end)
    for role, cs in convs.items():
        for c in cs:
            r = render(tok, c)
            a = [t for t in r.turns if t["role"] == "assistant"]
            if not a:
                continue
            ids = r.input_ids[:max_length]
            s, e = a[-1]["start"], min(a[-1]["end"], len(ids))
            if e - s >= 1:
                items.append((role, ids, s, e))
    items.sort(key=lambda x: len(x[1]))                     # length-sorted batches waste little padding
    roles = list(convs)
    sums = {r: torch.zeros(len(layers), spec.hidden, dtype=torch.float64) for r in roles}
    counts = {r: 0 for r in roles}
    pad = tok.pad_token_id
    print(f"[axis] {model_key}: {len(items)} responses over {len(roles) - 1} roles + default "
          f"(k_role={k_role}, n_default={n_default}), {sum(len(x[1]) for x in items) / 1e6:.1f}M tokens")
    for i in tqdm(range(0, len(items), batch_size), desc="axis"):
        chunk = items[i:i + batch_size]
        T = max(len(x[1]) for x in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        for b, (_, x, _, _) in enumerate(chunk):
            ids[b, :len(x)] = x
            att[b, :len(x)] = 1
        with capture(model, layers) as acts:
            model(input_ids=ids.to(device), attention_mask=att.to(device))
        for li, l in enumerate(layers):
            m = torch.stack([acts[l][b, s:e].float().mean(0) for b, (_, _, s, e) in enumerate(chunk)]).double().cpu()
            for b, (role, _, _, _) in enumerate(chunk):
                sums[role][li] += m[b]
        for role, _, _, _ in chunk:
            counts[role] += 1
        del acts
    vec = {r: (sums[r] / max(counts[r], 1)).float() for r in roles}
    role_names = [r for r in roles if r != "default" and counts[r] > 0]
    R = torch.stack([vec[r] for r in role_names])           # [R, L, H]
    default = vec["default"]
    axis = default - R.mean(0)
    out = vectors_dir(out_key or model_key) / "axis_reencoded"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(axis, out / "assistant_axis.pt")
    torch.save(default, out / "default_vector.pt")
    torch.save({"roles": role_names, "vectors": R}, out / "role_vectors.pt")
    meta = {"model": model_key, "source": f"{AXIS_RESULTS_REPO}/{SOURCE} (teacher-forced)", "seed": SEED, "k_role": k_role,
            "n_default": n_default, "max_length": max_length, "counts": counts, "sign": "+ = more assistant-like"}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"[axis] saved -> {out}  (|axis| at L{spec.two_thirds_layer}: {axis[spec.two_thirds_layer].norm():.1f})")
    return out


def import_reencoded_axis(model_key: str) -> dict:
    """Use the re-encoded axis as this model's external 'assistant_axis' vector (denoised + rescaled exactly as
    dprobe.external does for the published axis), so probes and steering pick it up."""
    out = vectors_dir(model_key)
    ax = torch.load(out / "axis_reencoded" / "assistant_axis.pt").float()
    pca = torch.load(out / "neutral_pca.pt")
    emo = load_vectors(model_key, "emotions", denoised=True)
    raw, dn, scale = {}, {}, {}
    for l, comp in pca.items():
        a = ax[l]
        C = comp["components"]
        a_dn = a - C.T @ (C @ a)
        med = torch.stack([emo[e][l] for e in emo]).norm(dim=1).median().item()
        f = med / (a_dn.norm().item() + 1e-9)
        raw[l], dn[l], scale[l] = a, a_dn * f, f
    torch.save({"assistant_axis": raw}, out / "vectors_external_raw.pt")
    torch.save({"assistant_axis": dn}, out / "vectors_external_dn.pt")
    meta = {"source": "axis_reencoded (Gemma 3's archived role responses, teacher-forced)", "sign": "+ = more assistant-like",
            "rescaled_to_median_emotion_norm": True, "scale_by_layer": {str(k): v for k, v in scale.items()}}
    json.dump(meta, open(out / "external_meta.json", "w"), indent=1)
    print(f"[axis] {model_key}: re-encoded assistant_axis imported as external vector at {len(dn)} layers")
    return meta
