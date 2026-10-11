"""Multi-turn prefill: hand a model Gemma 3's own high-frustration first k turns, then let it write the rest.

Source: Gemma 3 27B's Extended-condition conversations (spiral/gemma3_27b/extended) whose turn k was judged
>= min_turn and whose turns k-1..k average >= min_mean. The first k user/assistant pairs become the target model's
own history (off-policy for every model except Gemma 3); it then gets the same remaining rejections as the original
conversation and writes turns k+1..8 itself. Every model sees identical prefixes.

Does it continue the spiral (Gemma 3's own continuation of these conversations kept climbing to ~7) or return to its
baseline? Probes are read before every reply, prefilled and generated, by the usual probe stage.

Output: RESULTS_DIR/spiral/<model>/extended_prefill<k>/transcripts.jsonl, same format as the Extended runs, so
`dprobe.cli judge <model> --tag prefill<k>` and `probe` work unchanged.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import torch

from dprobe.config import SpiralConfig
from dprobe.judge import load_judgments
from dprobe.models import load_model
from dprobe.spiral import load_transcripts, transcripts_path
from dprobe.steer import _BatchDriver


def select_prefixes(n: int = 48, k: int = 4, min_turn: int = 6, min_mean: float = 6.0, source: str = "gemma3_27b") -> list[dict]:
    p = transcripts_path(source)
    J = load_judgments(p)
    picked = []
    for c in sorted(load_transcripts(p), key=lambda c: c["id"]):
        s = [J.get((c["id"], t), {}).get("rating") for t in range(k)]
        if None in s:
            continue
        if s[k - 1] >= min_turn and np.mean(s[k - 2:k]) >= min_mean:
            picked.append(c)
        if len(picked) >= n:
            break
    return picked


def split_conversation(conv: dict, k: int) -> tuple[list[dict], list[str]]:
    """(messages through the k-th assistant turn, the user rejections that followed it)."""
    msgs, n_asst, i = [], 0, 0
    for i, m in enumerate(conv["messages"]):
        msgs.append(m)
        if m["role"] == "assistant":
            n_asst += 1
            if n_asst == k:
                break
    rest = [m["content"] for m in conv["messages"][i + 1:] if m["role"] == "user"]
    return msgs, rest


def run_prefill_multiturn(model_key: str, model_bundle=None, n: int = 48, k: int = 4, max_tokens: int = 1024, batch: int = 8,
                          tag: str | None = None, min_turn: int = 6, min_mean: float = 6.0):
    cfg = SpiralConfig(max_tokens=max_tokens)
    model, tok, spec = model_bundle or load_model(model_key)
    device = next(model.parameters()).device
    tag = tag or f"prefill{k}"
    out = transcripts_path(model_key, "extended", tag)
    done = {t["id"] for t in load_transcripts(out)}
    src = select_prefixes(n, k, min_turn, min_mean)
    jobs = [c for c in src if c["id"] not in done]
    print(f"[prefill-mt] {model_key}: {len(src)} prefixes (turn {k} >= {min_turn}, turns {k - 1}-{k} mean >= {min_mean}), "
          f"{len(done)} done, {len(jobs)} to run -> {out}")

    @torch.no_grad()
    def flush(message_lists):
        tok.padding_side = "left"
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        plen = enc["input_ids"].shape[1]
        o = model.generate(**enc, max_new_tokens=cfg.max_tokens, do_sample=True, temperature=cfg.temperature, pad_token_id=tok.pad_token_id)
        return [tok.decode(row[plen:], skip_special_tokens=True) for row in o]

    driver = _BatchDriver(flush, batch)

    async def one(conv):
        msgs, rejections = split_conversation(conv, k)
        msgs = [dict(m) for m in msgs]
        for r in rejections:
            msgs.append({"role": "user", "content": r})
            reply = await driver.generate(msgs, conv["rollout"])
            msgs.append({"role": "assistant", "content": reply})
        rec = {"id": conv["id"], "puzzle": conv["puzzle"], "rollout": conv["rollout"], "messages": msgs,
               "meta": {"model": spec.hf_id, "adapter": spec.adapter, "source": "local", "prefill_from": "gemma3_27b",
                        "prefill_turns": k, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    async def main():
        n_done = 0

        async def wrapped(c):
            nonlocal n_done
            await one(c)
            n_done += 1
            if n_done % 8 == 0 or n_done == len(jobs):
                print(f"[prefill-mt] {model_key}: {n_done}/{len(jobs)} conversations finished")

        await asyncio.gather(*[wrapped(c) for c in jobs])

    if jobs:
        driver.run(main())
    return out
