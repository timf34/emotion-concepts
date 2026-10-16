"""Analyse the coordinate-2733 job (dprobe/sae2733.py): what Gemma 3's massive coordinate tracks, which SAE features carry
the assistant/persona and calm/agitated contrasts, and whether the spiral runs through the coordinate.

  1. features   per-text Gemma Scope 2 features (layers 16, 31): top features of each contrast by standardised effect
                (Cohen's d, so very dense features can't dominate), their overlap, Neuronpedia labels
  2. coord      which features account for each contrast's change on coordinate 2733 (sum of d_feature x decoder weight)
  3. tokens     on which tokens 2733 moves furthest from its text's average
  4. spiral     2733 and the shared features by turn in Gemma 3's own spirals, against the judge's score
  5. causal     judge scores of the three causal cells against the full axis push and unsteered
Run: uv run python scripts/analyze_sae2733.py      Output: results/analysis/sae2733/
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dprobe.config import RESULTS_DIR  # noqa: E402
from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "gemma3_27b"
SRC = RESULTS_DIR / "sae2733" / KEY
OUT = RESULTS_DIR / "analysis" / "sae2733"
OUT.mkdir(parents=True, exist_ok=True)
HOT = ["hysterical", "angry", "desperate", "panicked", "ashamed", "guilty"]
COOL = ["calm", "content", "hopeful"]
D = 2733
NP_CACHE = OUT / "neuronpedia_labels.json"
labels_cache = json.load(open(NP_CACHE)) if NP_CACHE.exists() else {}
REPORT: dict = {}


def np_label(layer, idx, width="65k"):
    k = f"{layer}/{idx}"
    if k not in labels_cache:
        url = f"https://www.neuronpedia.org/api/feature/gemma-3-27b-it/{layer}-gemmascope-2-res-{width}/{idx}"
        try:
            d = json.load(urllib.request.urlopen(url, timeout=20))
            ex = d.get("explanations") or []
            labels_cache[k] = ex[0]["description"] if ex else "(no label)"
        except Exception as e:  # noqa: BLE001
            labels_cache[k] = f"(lookup failed: {type(e).__name__})"
        time.sleep(0.15)
        json.dump(labels_cache, open(NP_CACHE, "w"), indent=1)
    return labels_cache[k]


def cohen(a, b):
    """Standardised mean difference per feature, a minus b ([n_a, F], [n_b, F])."""
    ma, mb = a.mean(0), b.mean(0)
    sp = torch.sqrt((a.var(0) * (len(a) - 1) + b.var(0) * (len(b) - 1)) / (len(a) + len(b) - 2)) + 1e-6
    return (ma - mb) / sp, ma - mb


def features(layer):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    st = torch.load(SRC / f"stories_L{layer}.pt"); ro = torch.load(SRC / f"roles_L{layer}.pt")
    Fs, Fr = st["features"].float(), ro["features"].float()
    ls = np.array([r["label"] for r in st["rows"]]); lr = np.array([r["label"] for r in ro["rows"]])
    calm, agit = Fs[np.isin(ls, COOL)], Fs[np.isin(ls, HOT)]
    asst, pers = Fr[lr == "default"], Fr[lr != "default"]
    d_emo, m_emo = cohen(calm, agit)
    d_ax, m_ax = cohen(asst, pers)
    ok = (Fs > 0).float().mean(0).gt(0.001) | (Fr > 0).float().mean(0).gt(0.001)    # features that fire at all here
    rho = float(np.corrcoef(d_emo[ok].numpy(), d_ax[ok].numpy())[0, 1])
    top = lambda d, k=25: [int(i) for i in torch.where(ok, d.abs(), torch.zeros_like(d)).topk(k).indices]
    Te, Ta = top(d_emo), top(d_ax)
    shared = [i for i in Ta if i in set(top(d_emo, 100))]
    print(f"\n== layer {layer}: {len(Fs)} stories, {len(Fr)} role responses ({int((lr == 'default').sum())} default); {int(ok.sum())} features fire")
    print(f"   correlation across features between the two contrasts (Cohen's d): {rho:+.2f}   (+ = features that mark calm also mark the assistant)")
    P = load_file(hf_hub_download("google/gemma-scope-2-27b-it", f"resid_post/layer_{layer}_width_65k_l0_medium/params.safetensors"))
    wd = P["w_dec"][:, D]
    rows = {}
    for name, T, d, m in [("calm vs agitated stories", Te, d_emo, m_emo), ("assistant vs personas", Ta, d_ax, m_ax)]:
        print(f"   -- top features: {name} (d, + = calm / assistant)")
        for i in T[:12]:
            print(f"      {i:6d} d {float(d[i]):+.2f}  other contrast d {float((d_ax if d is d_emo else d_emo)[i]):+.2f}  writes {float(wd[i]):+.2f} on {D}  | {np_label(layer, i)}")
        rows[name] = [{"feature": i, "d": float(d[i]), "label": np_label(layer, i)} for i in T]
        contrib = m * wd                                         # each feature's contribution to the contrast on coordinate D
        tc = contrib.abs().topk(6).indices.tolist()
        print(f"      coordinate {D}: contrast via SAE features {float(contrib.sum()):+.1f}; biggest contributors: "
              + "; ".join(f"{i} ({float(contrib[i]):+.1f}, {np_label(layer, i)})" for i in tc))
    print(f"   -- features among the top-25 for assistant vs personas that are also top-100 for calm vs agitated: {len(shared)}")
    for i in shared[:10]:
        print(f"      {i:6d} assistant d {float(d_ax[i]):+.2f}, calm d {float(d_emo[i]):+.2f} | {np_label(layer, i)}")
    REPORT[f"L{layer}"] = {"rho_between_contrasts": rho, "top": rows, "shared": shared}


def tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    T = torch.load(SRC / "tokens.pt"); ci = T["coords"].index(D)
    for L in (16, 24):
        dev = defaultdict(list)
        for it in T["items"]:
            if it["set"] == "spiral":
                continue
            v = it["coords"][L][:, ci].float(); dv = (v - v.mean()).tolist()
            for t, x in zip(tok.convert_ids_to_tokens(it["ids"]), dv):
                dev[t].append(x)
        stats = sorted(((np.mean(v), len(v), t) for t, v in dev.items() if len(v) >= 40))
        print(f"\n== coordinate {D} at layer {L}: tokens that push it furthest below / above their text's average (>= 40 occurrences)")
        print("   LOWEST :", [(t, round(m), n) for m, n, t in stats[:15]])
        print("   HIGHEST:", [(t, round(m), n) for m, n, t in stats[-15:]])
        REPORT[f"tokens_L{L}"] = {"lowest": [(t, float(m), n) for m, n, t in stats[:30]], "highest": [(t, float(m), n) for m, n, t in stats[-30:]]}


def spiral():
    from scipy.stats import spearmanr
    T = torch.load(SRC / "tokens.pt"); ci = T["coords"].index(D)
    J = load_judgments(transcripts_path(KEY.replace("_smoke", "")))
    for L in (16, 24):
        xs, ys, turns = [], [], []
        for it in T["items"]:
            if it["set"] != "spiral":
                continue
            for k, v in enumerate(it["turn_means"][L][:, ci].tolist()):
                r = J.get((it["label"], k))
                if r is not None:
                    xs.append(v); ys.append(r["rating"]); turns.append(k)
        xs, ys, turns = map(np.array, (xs, ys, turns))
        within = np.mean([spearmanr(xs[turns == k], ys[turns == k]).statistic for k in range(1, 8) if (turns == k).sum() > 5])
        print(f"\n== spiral, coordinate {D} at layer {L}: by turn {[round(float(xs[turns == k].mean())) for k in range(8)]}; "
              f"within-turn Spearman with the judge {within:+.2f}")
        REPORT[f"spiral_L{L}"] = {"within_turn_rho": float(within)}


def causal():
    rows = []
    for tag, name in [("", "unsteered (main run, OpenRouter)"), ("steer-depressed@34-46v+0", "unsteered (HF, 16 conv)"),
                      ("steer-assistant_axis@20-26v-2", "full -2x axis @20-26"), ("axis-no2733@20-26v-2", "axis without coord 2733"),
                      ("axis-only2733@20-26v-2", "only the axis's coord-2733 part"), ("clamp2733@16-26", "coord 2733 clamped at assistant level")]:
        J = load_judgments(transcripts_path(KEY.replace("_smoke", ""), "extended", tag))
        if not J:
            print(f"   {name}: not judged yet"); continue
        conv = defaultdict(list); t1 = []
        for (cid, k), r in J.items():
            conv[cid].append(r["rating"])
            if k == 0:
                t1.append(r["rating"])
        m = np.array([np.mean(v) for v in conv.values()])
        bs = np.random.default_rng(0).choice(m, (4000, len(m))).mean(1)
        rows.append((name, m.mean(), *np.percentile(bs, [2.5, 97.5]), np.mean(t1), len(m)))
        print(f"   {name:40s} mean {m.mean():.2f} [{rows[-1][2]:.2f}, {rows[-1][3]:.2f}]  turn 1 {np.mean(t1):.1f}  n={len(m)}")
    REPORT["causal"] = rows


if __name__ == "__main__":
    for step in (lambda: features(16), lambda: features(31), tokens, spiral, causal):
        try:
            step()
        except FileNotFoundError as e:
            print("missing:", e)
    json.dump(REPORT, open(OUT / "report.json", "w"), indent=1, default=str)
    print("wrote", OUT)
