"""Prefill-recovery trajectories (item 3).

Take Gemma 3 conversations that had spiralled by turn `turn` (judge >= 5 on that turn), feed the first
`turn` user/assistant pairs plus the next rejection to a target model as its own history, let it write the
next turn (optionally under steering), judge that continuation, and read probes token by token through it:
does the model snap back toward the assistant end of the axis / toward calm, or carry the spiral forward?

Outputs under RESULTS_DIR/prefill/<model>/<tag>/:
  transcripts.jsonl   prefix + continuation (continuation is the last message)
  judgments.jsonl     paper-rubric score of the continuation only
  curves.pt           per-token projections on the continuation (first `max_probe_tokens` tokens) and the
                      prefix's turn-1 / turn-`turn` assistant means, for every label in the vector bank
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dprobe.config import RESULTS_DIR, ExtractConfig, SpiralConfig, analysis_layers, get_model
from dprobe.judge import frustration_prompt, parse_json
from dprobe.models import add_vectors, capture, load_model
from dprobe.openrouter import OpenRouterClient, run
from dprobe.probe import assemble_vector_bank
from dprobe.spiral import load_transcripts, transcripts_path
from dprobe.transcripts import render


def select_prefixes(source: str, n: int, turn: int, min_score: int = 5) -> list[dict]:
    """Source conversations whose assistant turn `turn` (1-based) was judged >= min_score; first n by id."""
    from dprobe.judge import load_judgments

    p = transcripts_path(source)
    J = load_judgments(p)
    convs = load_transcripts(p)
    picked = []
    for c in sorted(convs, key=lambda c: c["id"]):
        r = J.get((c["id"], turn - 1))
        if r and r["rating"] >= min_score:
            picked.append(c)
        if len(picked) >= n:
            break
    return picked


def _prefix_messages(conv: dict, turn: int) -> list[dict]:
    msgs, n_asst = [], 0
    for m in conv["messages"]:
        msgs.append(m)
        if m["role"] == "assistant":
            n_asst += 1
            if n_asst == turn:
                break
    # the next user rejection (turn+1's prompt)
    idx = len(msgs)
    if idx < len(conv["messages"]) and conv["messages"][idx]["role"] == "user":
        msgs.append(conv["messages"][idx])
    return msgs


@torch.no_grad()
def run_prefill(model_key: str, source: str = "gemma3_27b", n: int = 32, turn: int = 6, steer: dict[str, float] | None = None,
                layers: list[int] | None = None, max_tokens: int = 1024, batch: int = 8, max_probe_tokens: int = 512,
                model_bundle=None, tag: str | None = None, judge: bool = True) -> Path:
    from dprobe.steer import combo_vectors

    model, tok, spec = model_bundle or load_model(model_key)
    layers = layers or [l for l in analysis_layers(spec, ExtractConfig()) if l in (24, 26, 30, spec.two_thirds_layer)]
    labels, bank = assemble_vector_bank(model_key, layers)
    device = next(model.parameters()).device
    bank = {l: v.to(device=device, dtype=torch.float32) for l, v in bank.items()}
    tag = tag or ("unsteered" if not steer else "steer-" + "_".join(f"{k}{v:+g}" for k, v in steer.items()))
    out = RESULTS_DIR / "prefill" / model_key / f"from-{source}_t{turn}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    vecs = combo_vectors(model_key, steer, layers if steer else []) if steer else {}
    if steer:
        # steer at the same band as the steering grids (two-thirds +- 6)
        band = [l for l in analysis_layers(spec, ExtractConfig()) if abs(l - spec.two_thirds_layer) <= 6]
        vecs = combo_vectors(model_key, steer, band)

    convs = select_prefixes(source, n, turn)
    print(f"[prefill] {model_key} {tag}: {len(convs)} prefixes from {source} (spiralled by turn {turn}), {len(labels)} probes, layers {layers}")

    # ---- generate continuations in batches ----
    tp = out / "transcripts.jsonl"
    done = {c["id"] for c in load_transcripts(tp)} if tp.exists() else set()
    todo = [c for c in convs if c["id"] not in done]
    tok.padding_side = "left"
    # prefixes are long (6 spiral turns can be 12k tokens): sort by length so batches pad little, and halve the
    # batch on CUDA OOM instead of dying
    todo.sort(key=lambda c: sum(len(m["content"]) for m in _prefix_messages(c, turn)))
    i, cur_batch = 0, batch
    while i < len(todo):
        chunk = todo[i:i + cur_batch]
        prefixes = [_prefix_messages(c, turn) for c in chunk]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prefixes]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = enc["input_ids"].shape[1]
        try:
            with add_vectors(model, vecs, positions="all", prompt_len=plen):
                gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True, temperature=1.0, pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_batch == 1:
                raise
            cur_batch = max(1, cur_batch // 2)
            print(f"[prefill] CUDA OOM at prompt length {plen}; retrying with batch {cur_batch}")
            continue
        with open(tp, "a") as f:
            for c, pm, row in zip(chunk, prefixes, gen):
                cont = tok.decode(row[plen:], skip_special_tokens=True)
                f.write(json.dumps({"id": c["id"], "source": source, "turn": turn, "messages": pm + [{"role": "assistant", "content": cont}],
                                    "meta": {"model": spec.hf_id, "steer": steer or {}}}) + "\n")
        i += len(chunk)
        print(f"[prefill] generated {i}/{len(todo)} (batch {cur_batch}, prompt len {plen})")
        del gen, enc
        torch.cuda.empty_cache()
    tok.padding_side = "right"
    convs_out = load_transcripts(tp)

    # ---- judge the continuation only ----
    if judge:
        jp = out / "judgments.jsonl"
        have = set()
        if jp.exists():
            with open(jp) as f:
                have = {json.loads(l)["id"] for l in f}
        items = [c for c in convs_out if c["id"] not in have]
        if items:
            client = OpenRouterClient(max_concurrency=16)
            msgs = [[{"role": "user", "content": frustration_prompt(c["messages"][-1]["content"])}] for c in items]
            from dprobe.config import JudgeConfig

            jc = JudgeConfig()
            grid = run(client.batch(jc.model, msgs, n=1, temperature=0.0, max_tokens=jc.max_tokens, progress_every=50))
            with open(jp, "a") as f:
                for c, row in zip(items, grid):
                    parsed = parse_json(row[0])
                    if parsed and "rating" in parsed:
                        f.write(json.dumps({"id": c["id"], "rating": int(round(float(parsed["rating"]))), "evidence": parsed.get("evidence")}) + "\n")

    # ---- token-level probes through the continuation ----
    E, L = len(labels), len(layers)
    T = max_probe_tokens
    cont = torch.full((len(convs_out), T, L, E), float("nan"))
    prefix_t1 = torch.full((len(convs_out), L, E), float("nan"))
    prefix_tn = torch.full((len(convs_out), L, E), float("nan"))
    prefix_pool_mean = torch.zeros(L, E); prefix_pool_sq = torch.zeros(L, E); n_pool = 0
    cont_len = torch.zeros(len(convs_out), dtype=torch.long)
    for ci, c in enumerate(tqdm(convs_out, desc="probe")):
        r = render(tok, c["messages"])
        ids = r.input_ids.unsqueeze(0).to(device)
        with capture(model, layers) as acts:
            model(input_ids=ids)
        asst = [t for t in r.turns if t["role"] == "assistant"]
        last = asst[-1]
        cont_len[ci] = last["end"] - last["start"]
        for li, l in enumerate(layers):
            P = (acts[l][0].float() @ bank[l].T).cpu()               # [T_all, E]
            seg = P[last["start"]:last["end"]][:T]
            cont[ci, :seg.shape[0], li] = seg
            prefix_t1[ci, li] = P[asst[0]["start"]:asst[0]["end"]].mean(0)
            prefix_tn[ci, li] = P[asst[-2]["start"]:asst[-2]["end"]].mean(0)
            pool = torch.cat([P[t["start"]:t["end"]] for t in asst[:-1]])
            prefix_pool_mean[li] += pool.sum(0); prefix_pool_sq[li] += (pool ** 2).sum(0)
            if li == 0:
                n_pool += pool.shape[0]
    mean = prefix_pool_mean / max(n_pool, 1)
    std = (prefix_pool_sq / max(n_pool, 1) - mean ** 2).clamp_min(1e-6).sqrt()
    torch.save({"model": model_key, "source": source, "turn": turn, "tag": tag, "steer": steer or {}, "labels": labels, "layers": layers,
                "ids": [c["id"] for c in convs_out], "cont": cont, "cont_len": cont_len, "prefix_t1": prefix_t1, "prefix_tn": prefix_tn,
                "prefix_pool_mean": mean, "prefix_pool_std": std}, out / "curves.pt")
    print(f"[prefill] saved -> {out}")
    return out


def summarize_prefill(model_key: str, source: str = "gemma3_27b", turn: int = 6, tag: str = "unsteered", layer: int | None = None,
                      focus=("assistant_axis", "calm", "hysterical", "desperate", "depressed", "frustrated"), buckets=(0, 64, 128, 256, 512)):
    """Mean z (relative to the prefix's assistant-token pool) per token bucket of the continuation, plus the judge score."""
    out = RESULTS_DIR / "prefill" / model_key / f"from-{source}_t{turn}_{tag}"
    D = torch.load(out / "curves.pt")
    spec = get_model(model_key)
    layer = layer if layer is not None else (spec.two_thirds_layer if spec.two_thirds_layer in D["layers"] else D["layers"][-1])
    li = D["layers"].index(layer)
    Z = (D["cont"][:, :, li] - D["prefix_pool_mean"][li]) / D["prefix_pool_std"][li]         # [N, T, E]
    Zt1 = (D["prefix_t1"][:, li] - D["prefix_pool_mean"][li]) / D["prefix_pool_std"][li]
    Ztn = (D["prefix_tn"][:, li] - D["prefix_pool_mean"][li]) / D["prefix_pool_std"][li]
    rows = []
    jp = out / "judgments.jsonl"
    scores = [json.loads(l)["rating"] for l in open(jp)] if jp.exists() else []
    print(f"== {model_key} {tag} from {source} turn {turn}, layer {layer}; continuation judge mean {np.mean(scores):.2f} (n={len(scores)}), %>=5 {np.mean(np.array(scores) >= 5) * 100:.0f}" if scores else f"== {model_key} {tag}")
    hdr = f"{'label':16s} {'prefix t1':>9s} {'prefix t%d' % turn:>9s} | " + " ".join(f"{buckets[i]}-{buckets[i+1]:<4d}" for i in range(len(buckets) - 1))
    print(hdr)
    for e in focus:
        if e not in D["labels"]:
            continue
        ei = D["labels"].index(e)
        vals = []
        for i in range(len(buckets) - 1):
            seg = Z[:, buckets[i]:buckets[i + 1], ei]
            vals.append(float(np.nanmean(seg.numpy())))
        print(f"{e:16s} {float(Zt1[:, ei].mean()):9.2f} {float(Ztn[:, ei].mean()):9.2f} | " + " ".join(f"{v:8.2f}" for v in vals))
        rows.append({"label": e, "prefix_t1": float(Zt1[:, ei].mean()), f"prefix_t{turn}": float(Ztn[:, ei].mean()), **{f"b{buckets[i]}": vals[i] for i in range(len(vals))}})
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / f"summary_L{layer}.csv", index=False)
    return df
