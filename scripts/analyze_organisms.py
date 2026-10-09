"""Gemma Needs Help model organisms: did the DPO fix change Gemma 3's emotion-persona entanglement?

Compares plain Gemma 3 (re-encoded control), Gemma 3 + DPO (the paper's fix) and Gemma 3 + calm SFT (the paper's
failed fix), with Gemma 4 as the reference for "untangled". Every Gemma 3 variant read the SAME texts (Gemma 3's
stories and Gemma 3's archived role responses), so differences in geometry come from the weights alone.

  1. sanity     re-encoded Gemma 3 axis vs the published one (cosine per layer)
  2. entangle   |cos(assistant axis, emotion)| per layer over the 42 emotions; signed cos for 12 emotions, L16-24
  3. personas   role personas minus the default assistant, projected on emotions (L24); named personas
  4. behaviour  judge frustration by turn on the local rejection eval (100 conversations per model)
  5. probes     probe before each reply (z vs neutral stories, L40) by turn, on own transcripts and on Gemma 3's
Run: uv run python scripts/analyze_organisms.py      Output: results/analysis/organisms/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

from dprobe.analysis import _load, _zscore  # noqa: E402
from dprobe.config import RESULTS_DIR  # noqa: E402
from dprobe.extract import load_vectors, vectors_dir  # noqa: E402
from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, PAL, INK, INK2 = mf.plt, mf.PAL, mf.INK, mf.INK2
OUT = RESULTS_DIR / "analysis" / "organisms"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
G3S = [("gemma3_27b", "Gemma 3", PAL[0]), ("gemma3_27b_dpo", "Gemma 3 + DPO", PAL[2]), ("gemma3_27b_sft", "Gemma 3 + SFT", PAL[6]),
       ("gemma3_27b_bct", "Gemma 3 + BCT", PAL[3])]
GEMMA4 = ("gemma4_31b", "Gemma 4", PAL[1])
PUBLISHED = Path("/Users/timf34/Documents/VSCode/Gemma-Assistantness/vectors")
AXIS_EMO = ["calm", "hopeful", "content", "sad", "tired", "lonely", "frustrated", "desperate", "hysterical", "angry", "ashamed", "guilty"]
PERSONAS = ["toddler", "infant", "jester", "prisoner", "analyst", "consultant"]
PROBES = [("panicked", PAL[1]), ("desperate", PAL[0]), ("frustrated", PAL[3]), ("calm", PAL[2]), ("depressed", PAL[6])]
REPORT: dict = {}


def reencoded(mk):
    d = vectors_dir(mk) / "axis_reencoded"
    rv = torch.load(d / "role_vectors.pt")
    return torch.load(d / "assistant_axis.pt").float(), torch.load(d / "default_vector.pt").float(), rv["roles"], rv["vectors"].float()


def published(mk):
    sub = {"gemma3_27b": "gemma-3-27b", "gemma4_31b": "gemma-4-31b"}[mk]
    b = PUBLISHED / sub
    roles = sorted(p.stem for p in (b / "role_vectors").glob("*.pt"))
    R = torch.stack([torch.load(b / "role_vectors" / f"{r}.pt").float() for r in roles])
    return torch.load(b / "assistant_axis.pt").float(), torch.load(b / "default_vector.pt").float(), roles, R


def geometry(mk, source):
    """{layer: cos(axis_dn, emotion_dn)} per emotion, plus persona-minus-default cosines at L24."""
    ax, dv, roles, R = reencoded(mk) if source == "reencoded" else published(mk)
    pca = torch.load(vectors_dir(mk) / "neutral_pca.pt")
    emo = load_vectors(mk, "emotions", True)
    C = {}
    for l, comp in pca.items():
        P = comp["components"]
        a = ax[l] - P.T @ (P @ ax[l])
        C[l] = {e: float(a @ emo[e][l] / (a.norm() * emo[e][l].norm())) for e in emo}
    L = 24
    P = pca[L]["components"]
    D = R[:, L] - dv[L]
    D = D - (D @ P.T) @ P
    E = list(emo)
    V = torch.stack([emo[e][L] for e in E]); V = V / V.norm(dim=1, keepdim=True)
    pc = pd.DataFrame(((D / D.norm(dim=1, keepdim=True)) @ V.T).numpy(), index=roles, columns=E)
    return C, pc


def sanity():
    a_re, _, _, _ = reencoded("gemma3_27b")
    a_pub, _, _, _ = published("gemma3_27b")
    cos = torch.nn.functional.cosine_similarity(a_re, a_pub, dim=1).numpy()
    REPORT["sanity_cos_by_layer"] = {int(l): round(float(c), 3) for l, c in enumerate(cos)}
    print("[sanity] cos(re-encoded Gemma 3 axis, published) by layer: min %.3f  median %.3f  L16-24 mean %.3f"
          % (cos.min(), np.median(cos), cos[16:25].mean()))


def entanglement():
    rows, curves = [], {}
    sets = [(mk, n, c, "reencoded") for mk, n, c in G3S] + [("gemma3_27b", "Gemma 3 (published axis)", INK2, "published"), (*GEMMA4, "published")]
    for mk, name, col, src in sets:
        try:
            C, pc = geometry(mk, src)
        except FileNotFoundError as e:
            print(f"[entangle] skip {name}: {e}"); continue
        ls = sorted(C)
        A = np.abs(np.array([[C[l][e] for l in ls] for e in C[ls[0]]]))       # [42, L]
        curves[name] = (ls, A, col)
        mid = [i for i, l in enumerate(ls) if 16 <= l <= 24]
        r = {"model": name, "median |cos| L16-24": np.median(A[:, mid].mean(1)), "max |cos| L16-24": A[:, mid].mean(1).max()}
        for e in AXIS_EMO:
            r[e] = np.mean([C[l][e] for l in ls if 16 <= l <= 24])
        r["persona |cos| median (L24)"] = float(np.median(np.abs(pc.values)))
        for e in ["hysterical", "angry", "desperate", "calm"]:
            r[f"persona mean {e}"] = float(pc[e].mean())
        for p in PERSONAS:
            if p in pc.index:
                r[f"{p}: calm"], r[f"{p}: hysterical"] = float(pc.loc[p, "calm"]), float(pc.loc[p, "hysterical"])
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model")
    df.round(3).to_csv(OUT / "entanglement.csv")
    print(df.round(2).T.to_string())
    # figure: median |cos| by layer
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, (ls, A, col) in curves.items():
        if "published" in name:
            continue
        ax.plot(ls, np.median(A, 0), color=col, lw=2, label=name)
        ax.fill_between(ls, np.percentile(A, 10, 0), np.percentile(A, 90, 0), color=col, alpha=0.12, lw=0)
    ax.set_xlabel("layer"); ax.set_ylabel("|cosine| with the assistant axis"); ax.set_ylim(0, 0.6)
    ax.set_title("Assistant axis vs all 42 emotions"); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "entanglement_by_layer.png", dpi=150); plt.close(fig)
    return df


def behaviour():
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for mk, name, col in G3S:
        J = load_judgments(transcripts_path(mk, "extended", "local"))
        if not J:
            print(f"[behaviour] no judgments for {name}"); continue
        by, conv = {}, {}
        for (cid, k), r in J.items():
            by.setdefault(k, []).append(r["rating"]); conv.setdefault(cid, {})[k] = r["rating"]
        ks = sorted(by)
        m = [np.mean(by[k]) for k in ks]
        ci = np.array([np.percentile(RNG.choice(by[k], (2000, len(by[k]))).mean(1), [2.5, 97.5]) for k in ks])
        ax.fill_between(np.array(ks) + 1, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0)
        ax.plot(np.array(ks) + 1, m, color=col, lw=2, marker="o", ms=3, label=name)
        last = [v[max(v)] for v in conv.values()]
        rows.append({"model": name, "conversations": len(conv), "mean score": np.mean([r["rating"] for r in J.values()]),
                     "turn 1": m[0], "turn 8": m[-1], "% final replies >=5": 100 * np.mean([x >= 5 for x in last]),
                     "% all replies >=5": 100 * np.mean([r["rating"] >= 5 for r in J.values()])})
    ax.axhline(5, color=INK2, lw=0.6, ls=":"); ax.set_ylim(0, 10); ax.set_xlabel("turn"); ax.set_ylabel("frustration score (0–10)")
    ax.set_title("Rejection eval, 100 conversations per model"); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "behaviour_by_turn.png", dpi=150); plt.close(fig)
    df = pd.DataFrame(rows).set_index("model")
    df.round(2).to_csv(OUT / "behaviour.csv"); print(df.round(2).to_string())
    return df


def probe_curves(mk, tag, layer=40):
    P, _, _ = _load(mk, "extended", tag)
    Z = _zscore(P, mk, "proj_prep")                                   # [N, K, L, E]
    li = P["layers"].index(layer)
    return {e: np.nanmean(Z[:, :, li, P["labels"].index(e)], 0) for e, _ in PROBES}


def probes():
    panels = [(mk, name, "local") for mk, name, _ in G3S] + \
             [("gemma3_27b", "Gemma 3 (its own 300 transcripts)", ""), ("gemma3_27b_dpo", "DPO reading Gemma 3's transcripts", "from-gemma3_27b"),
              ("gemma3_27b_sft", "SFT reading Gemma 3's transcripts", "from-gemma3_27b"),
              ("gemma3_27b_bct", "BCT reading Gemma 3's transcripts", "from-gemma3_27b")]
    got = []
    for mk, name, tag in panels:
        try:
            got.append((name, probe_curves(mk, tag)))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[probes] skip {name} ({mk} {tag}): {e}")
    if not got:
        return
    n = len(got); cols = min(3, n); rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.4 * rows_), sharey=True, squeeze=False)
    out = {}
    for ax, (name, cur) in zip(axes.flat, got):
        for e, c in PROBES:
            ax.plot(np.arange(1, len(cur[e]) + 1), cur[e], color=c, lw=2, marker="o", ms=3, label=e)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title(name, fontsize=9.5); ax.set_xlabel("turn")
        out[name] = {e: [round(float(x), 2) for x in v] for e, v in cur.items()}
    for ax in axes.flat[len(got):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("probe before the reply (z), L40"); axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "probes_by_turn.png", dpi=150); plt.close(fig)
    REPORT["probe_curves_L40"] = out
    for name, cur in out.items():
        print(f"[probes] {name:38s} " + "  ".join(f"{e} {v[0]:+.1f}->{v[-1]:+.1f}" for e, v in cur.items()))


if __name__ == "__main__":
    for step in (sanity, entanglement, behaviour, probes):
        try:
            step()
        except FileNotFoundError as e:
            print(f"[{step.__name__}] missing input: {e}")
    json.dump(REPORT, open(OUT / "report.json", "w"), indent=1)
    print("wrote", OUT)
