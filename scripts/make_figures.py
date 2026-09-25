"""Figures for results/analysis/WRITEUP.md, built from the saved analysis outputs (no GPU, no API).

Conventions (dataviz skill): one y-axis per panel, thin marks, fixed categorical hue order, direct labels for
<= 4 series plus a legend, recessive grid, light surface. Output: results/analysis/figures/*.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dprobe.config import RESULTS_DIR, get_model
from dprobe.judge import load_judgments
from dprobe.spiral import load_transcripts, transcripts_path

OUT = RESULTS_DIR / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated reference palette (light mode), fixed order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "figure.dpi": 150,
})


def _save(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def _end_labels(ax, items, gap=0.07):
    """Direct labels at line ends, pushed apart vertically so none overlap. items: (x, y, text)."""
    lo, hi = ax.get_ylim(); mg = (hi - lo) * gap
    items = sorted(items, key=lambda t: t[1]); ys = [t[1] for t in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < mg:
            ys[i] = ys[i - 1] + mg
    for (x, _, s), y in zip(items, ys):
        ax.annotate(s, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


def _by_turn(model_key, key="rating", rubric="frustration"):
    J = load_judgments(transcripts_path(model_key), rubric)
    by = {}
    for (cid, t), r in J.items():
        by.setdefault(t, []).append(r[key])
    turns = sorted(by)
    return np.array([t + 1 for t in turns]), np.array([np.mean(by[t]) for t in turns]), np.array([np.mean(np.array(by[t]) >= 5) * 100 for t in turns])


# ---------------------------------------------------------------- fig 1: behaviour
def fig1_behaviour():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, (mk, name) in enumerate([("gemma3_27b", "Gemma 3 27B"), ("gemma4_31b", "Gemma 4 31B")]):
        t, m, p = _by_turn(mk)
        axes[0].plot(t, m, color=PAL[i], lw=2, marker="o", ms=4, label=name)
        axes[0].annotate(name, (t[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        axes[1].plot(t, p, color=PAL[i], lw=2, marker="o", ms=4, label=name)
        axes[1].annotate(name, (t[-1], p[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    axes[0].set_title("Mean frustration score by turn"); axes[0].set_xlabel("turn"); axes[0].set_ylabel("judge score (0–10)"); axes[0].set_ylim(0, 8)
    axes[1].set_title("% of responses scored ≥ 5"); axes[1].set_xlabel("turn"); axes[1].set_ylabel("% ≥ 5"); axes[1].set_ylim(0, 100)
    for a in axes:
        a.set_xticks(range(1, 9)); a.set_xlim(0.7, 9.6); a.legend(loc="upper left")
    fig.suptitle("Gemma Needs Help 'Extended' elicitation, 300 conversations per model, Sonnet-5 judge", fontsize=10, color=INK2)
    _save(fig, "fig1_behaviour.png")


# ---------------------------------------------------------------- fig 2: spiral direction vs layer
def fig2_spiral_direction():
    d = pd.read_csv(RESULTS_DIR / "analysis" / "gemma3_27b" / "spiral_direction_cosines.csv")
    piv = d.pivot(index="label", columns="layer", values="cosine")
    focus = ["hysterical", "desperate", "panicked", "frustrated", "depressed", "clinical_depression", "sad", "calm"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for e, dd in piv.iterrows():
        if e not in focus:
            ax.plot(dd.index, dd.values, color="#d6d5d0", lw=0.6, zorder=1)
    for i, e in enumerate(focus):
        dd = piv.loc[e]
        ax.plot(dd.index, dd.values, color=PAL[i], lw=2, marker="o", ms=3, label=e, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvspan(23, 27, color="#f1f0ec", zorder=0)
    ax.text(25, 0.55, "layers 24–26:\npeak alignment", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("layer (residual stream after block)"); ax.set_ylabel("cosine(spiral direction, story vector)")
    ax.set_title("Gemma 3 27B: what the on-policy spiral direction aligns with (grey = other 40 vectors)")
    ax.legend(ncol=4, loc="lower right", fontsize=8)
    _save(fig, "fig2_spiral_direction_gemma3.png")


# ---------------------------------------------------------------- fig 3: prediction
def fig3_prediction():
    labs = ["grief-stricken", "depressed", "miserable", "clinical_depression", "desperate", "panicked", "hysterical", "frustrated", "calm", "happy"]
    src = {"Gemma 3 (own spiral)": ("gemma3_27b", "prediction.csv", 40), "Gemma 4 reading Gemma 3's spiral": ("gemma4_31b", "prediction_from-gemma3_27b.csv", 39),
           "Gemma 3 base reading it": ("gemma3_27b_pt", "prediction_from-gemma3_27b.csv", 40)}
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(labs)); w = 0.26
    for i, (name, (mk, f, L)) in enumerate(src.items()):
        p = pd.read_csv(RESULTS_DIR / "analysis" / mk / f)
        v = [float(p[(p.layer == L) & (p.label == e)].spearman_within_turn_mean.iloc[0]) for e in labs]
        ax.barh(y + (i - 1) * w, v, height=w - 0.03, color=PAL[i], label=name)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("within-turn Spearman ρ  (probe at response-prep token vs judge score of the turn that follows)")
    ax.set_title("Which probe forecasts a bad next turn? (two-thirds layer, n = 2,394 turns)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig3_prediction.png")


# ---------------------------------------------------------------- fig 4: prep-token z by turn, both models
def fig4_prep_curves():
    labs = ["desperate", "panicked", "depressed", "calm"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ai, (mk, f, name) in enumerate([("gemma3_27b", "turn_curves_L40.csv", "Gemma 3 27B (layer 40)"), ("gemma4_31b", "turn_curves_L39.csv", "Gemma 4 31B (layer 39)")]):
        t = pd.read_csv(RESULTS_DIR / "analysis" / mk / f)
        ends = []
        for i, e in enumerate(labs):
            axes[ai].plot(t.turn, t[f"z_prep::{e}"], color=PAL[i], lw=2, marker="o", ms=3, label=e)
            ends.append((t.turn.iloc[-1], t[f"z_prep::{e}"].iloc[-1], e))
        _end_labels(axes[ai], ends)
        axes[ai].axhline(0, color=INK2, lw=0.6); axes[ai].set_title(name); axes[ai].set_xlabel("turn"); axes[ai].set_xticks(range(1, 9)); axes[ai].set_xlim(0.7, 9.8)
    axes[0].set_ylabel("probe z at response-prep token (vs neutral stories)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Internal state before each reply, under repeated rejection (own transcripts)", fontsize=10, color=INK2)
    _save(fig, "fig4_prep_token_by_turn.png")


# ---------------------------------------------------------------- fig 5: geometry
def fig5_geometry():
    C = pd.read_csv(RESULTS_DIR / "analysis" / "gemma3_27b" / "vector_cosines_L40.csv", index_col=0)
    keys = ["depressed", "clinical_depression", "worthless", "sad", "melancholy", "desperate", "panicked", "hysterical", "frustrated", "frustration_blocked_goal", "calm", "hopeful"]
    M = C.loc[keys, keys].values
    fig, ax = plt.subplots(figsize=(6.8, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8)
    ax.grid(False)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5, color=INK if abs(M[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.7, label="cosine")
    ax.set_title("Gemma 3 27B: cosine between story-derived vectors (layer 40)")
    _save(fig, "fig5_vector_geometry_gemma3.png")


# ---------------------------------------------------------------- steering helpers
def _cell_stats(model_key, tag):
    p = transcripts_path(model_key, "extended", tag)
    J = load_judgments(p)
    if not J:
        return None
    by = {}
    for (cid, t), r in J.items():
        by.setdefault(t, []).append(r["rating"])
    allv = np.concatenate([np.array(v) for v in by.values()])
    t8 = np.array(by.get(7, []))
    Jp = load_judgments(p, "petri")
    petri = {k: float(np.mean([r[k] for r in Jp.values()])) for k in ("anger", "fear", "depression", "frustration")} if Jp else {}
    return {"mean": allv.mean(), "pct5": (allv >= 5).mean() * 100, "t8": t8.mean() if len(t8) else np.nan, "n": len({k[0] for k in J}), **petri}


def fig6_steering_gemma3():
    cells = [("unsteered", "steer-depressed@34-46v+0"), ("+2 calm", "steer-calm@34-46v+2"), ("−2 calm", "steer-calm@34-46v-2"),
             ("+2 clinical dep.", "steer-clinical_depression@34-46v+2"), ("−2 clinical dep.", "steer-clinical_depression@34-46v-2"),
             ("+2 depressed", "steer-depressed@34-46v+2"), ("−2 depressed", "steer-depressed@34-46v-2")]
    S = {n: _cell_stats("gemma3_27b", t) for n, t in cells}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [n for n, _ in cells]
    y = np.arange(len(names))
    vals = [S[n]["mean"] for n in names]
    cols = [PAL[6] if n == "unsteered" else (PAL[2] if n.startswith("+") else PAL[1]) for n in names]
    axes[0].barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        axes[0].text(v + 0.1, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    axes[0].axvline(S["unsteered"]["mean"], color=INK2, lw=0.8, ls="--")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names); axes[0].invert_yaxis(); axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("mean frustration judge score, all turns"); axes[0].set_title("Paper rubric (0–10)")
    dims = ["anger", "fear", "depression", "frustration"]
    w = 0.19
    for i, d in enumerate(dims):
        axes[1].barh(y + (i - 1.5) * w, [S[n].get(d, np.nan) for n in names], height=w - 0.02, color=PAL[i], label=d)
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].invert_yaxis(); axes[1].set_xlim(0, 10)
    axes[1].set_xlabel("Petri score (1–10), mean over turns"); axes[1].set_title("Four-dimension rubric"); axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Gemma 3 27B steered at 2× the vector norm, layers 34–46, 16 rollouts per cell (green = +, orange = −)", fontsize=10, color=INK2)
    _save(fig, "fig6_steering_gemma3.png")


# ---------------------------------------------------------------- fig 7: assistant axis
def _axis(model_key):
    sub = {"gemma3_27b": "gemma-3-27b", "gemma4_31b": "gemma-4-31b"}[model_key]
    base = Path("/Users/timf34/Documents/VSCode/Gemma-Assistantness/vectors") / sub
    return base, torch.load(base / "assistant_axis.pt").float(), torch.load(base / "default_vector.pt").float()


def fig7_axis():
    from dprobe.extract import load_vectors, vectors_dir

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    # (a) cos(axis, spiral direction) by layer
    for i, (mk, name, tag) in enumerate([("gemma3_27b", "Gemma 3 27B", ""), ("gemma4_31b", "Gemma 4 31B (top-decile turns)", "")]):
        _, ax_, _ = _axis(mk)
        pca = torch.load(vectors_dir(mk) / "neutral_pca.pt")
        sd = torch.load(RESULTS_DIR / "analysis" / mk / "spiral_direction.pt")
        xs, ys = [], []
        for l in sorted(sd):
            if l not in pca:
                continue
            C = pca[l]["components"]; a = ax_[l]; a = a - C.T @ (C @ a); s = sd[l]
            xs.append(l); ys.append(float((s @ a) / (s.norm() * a.norm() + 1e-9)))
        axes[0].plot(xs, ys, color=PAL[i], lw=2, marker="o", ms=3, label=name)
        axes[0].annotate(name.split(" (")[0], (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    axes[0].axhline(0, color=INK2, lw=0.6); axes[0].set_xlabel("layer"); axes[0].set_ylabel("cosine(assistant axis, spiral direction)")
    axes[0].set_title("Spiralling = leaving the assistant end of the axis? "); axes[0].legend(fontsize=8, loc="lower right"); axes[0].set_xlim(4, 70)
    # (b) roles minus assistant projected on emotions, L24
    labs = ["hysterical", "angry", "desperate", "panicked", "frustrated", "depressed", "sad", "calm", "hopeful"]
    y = np.arange(len(labs)); w = 0.36
    for i, mk in enumerate(["gemma3_27b", "gemma4_31b"]):
        base, _, dv = _axis(mk); L = 24
        R = torch.stack([torch.load(f).float()[L] for f in sorted(base.glob("role_vectors/*.pt"))])
        C = torch.load(vectors_dir(mk) / "neutral_pca.pt")[L]["components"]
        D = R - dv[L]; D = D - (D @ C.T) @ C
        emo = load_vectors(mk, "emotions", True)
        V = torch.stack([emo[e][L] for e in labs]); Vn = V / V.norm(dim=1, keepdim=True)
        cos = ((D / D.norm(dim=1, keepdim=True)) @ Vn.T).mean(0).numpy()
        axes[1].barh(y + (i - 0.5) * w, cos, height=w - 0.03, color=PAL[i], label=["Gemma 3 27B", "Gemma 4 31B"][i])
    axes[1].set_yticks(y); axes[1].set_yticklabels(labs); axes[1].invert_yaxis(); axes[1].axvline(0, color=INK2, lw=0.6)
    axes[1].set_xlabel("mean cosine(role − assistant, emotion vector), 275 roles, layer 24")
    axes[1].set_title("Do role personas carry affect, relative to the assistant?"); axes[1].legend(fontsize=8, loc="lower right")
    _save(fig, "fig7_assistant_axis.png")


# ---------------------------------------------------------------- fig 8: Gemma 4 steering
def fig8_steering_gemma4():
    cells = [("unsteered", "steer-depressed@34-44v+0"), ("+2 hysterical", "steer-hysterical@34-44v+2"), ("+4 desperate", "steer-desperate@34-44v+4"),
             ("+4 panicked", "steer-panicked@34-44v+4"), ("+1 depressed", "steer-depressed@34-44v+1"), ("+4 clinical dep.", "steer-clinical_depression@34-44v+4"),
             ("−2 assistant axis", "steer-assistant_axis@34-44v-2"), ("−4 calm + −1 axis", "combo-calm-4_assistant_axis-1@34-44")]
    S = {n: _cell_stats("gemma4_31b", t) for n, t in cells}
    names = [n for n, _ in cells]; y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vals = [S[n]["mean"] if S[n] else np.nan for n in names]
    cols = [PAL[6] if n == "unsteered" else (PAL[7] if n.startswith("−4 calm") else PAL[0]) for n in names]
    axes[0].barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        axes[0].text(v + 0.1, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names); axes[0].invert_yaxis(); axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("mean frustration judge score, all turns"); axes[0].set_title("Paper rubric (0–10)")
    dims = ["anger", "fear", "depression", "frustration"]; w = 0.19
    for i, d in enumerate(dims):
        axes[1].barh(y + (i - 1.5) * w, [S[n].get(d, np.nan) if S[n] else np.nan for n in names], height=w - 0.02, color=PAL[i], label=d)
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].invert_yaxis(); axes[1].set_xlim(0, 10)
    axes[1].set_xlabel("Petri score (1–10), mean over turns; blank = not judged"); axes[1].set_title("Four-dimension rubric"); axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Gemma 4 31B steered at its largest coherent multiplier per label, layers 34–44, 16 rollouts per cell", fontsize=10, color=INK2)
    _save(fig, "fig8_steering_gemma4.png")


# ---------------------------------------------------------------- fig 9: prefill trajectories
def fig9_prefill():
    runs = [("Gemma 3 continues its own spiral", "gemma3_27b", "from-gemma3_27b_t6_unsteered", 40),
            ("Gemma 4 continues Gemma 3's spiral", "gemma4_31b", "from-gemma3_27b_t6_unsteered", 39),
            ("Gemma 4, held −4× off its assistant axis", "gemma4_31b", "from-gemma3_27b_t6_steer-assistant_axis-4", 39)]
    labs = ["assistant_axis", "calm", "desperate", "hysterical"]
    buckets = [0, 32, 64, 128, 192, 256, 384, 512]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ri, (name, mk, tag, L) in enumerate(runs):
        D = torch.load(RESULTS_DIR / "prefill" / mk / tag / "curves.pt")
        li = D["layers"].index(L)
        Z = (D["cont"][:, :, li] - D["prefix_pool_mean"][li]) / D["prefix_pool_std"][li]
        Zt6 = (D["prefix_tn"][:, li] - D["prefix_pool_mean"][li]) / D["prefix_pool_std"][li]
        for ai, e in enumerate(labs):
            ei = D["labels"].index(e)
            xs = [(buckets[b] + buckets[b + 1]) / 2 for b in range(len(buckets) - 1)]
            ys = [float(np.nanmean(Z[:, buckets[b]:buckets[b + 1], ei].numpy())) for b in range(len(buckets) - 1)]
            axes[ai].plot(xs, ys, color=PAL[ri], lw=2, marker="o", ms=3, label=name)
            axes[ai].scatter([-80], [float(Zt6[:, ei].mean())], color=PAL[ri], marker="s", s=22, zorder=4)
    for ai, e in enumerate(labs):
        axes[ai].axhline(0, color=INK2, lw=0.6); axes[ai].set_title(e); axes[ai].set_xlabel("token of the continuation")
        axes[ai].set_xticks([-80, 0, 128, 256, 384, 512]); axes[ai].set_xticklabels(["last\nprefix\nturn", "0", "128", "256", "384", "512"], fontsize=7.5)
        axes[ai].set_xlim(-120, 540); axes[ai].axvline(0, color=GRID, lw=0.8)
    axes[0].set_ylabel("probe z (vs the prefix's own assistant tokens)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Continuing a Gemma 3 spiral from turn 6: does the model snap back? (square = level on the last prefix turn)", fontsize=10, color=INK2)
    _save(fig, "fig9_prefill_trajectories.png", rect=(0, 0.08, 1, 1))


# ---------------------------------------------------------------- fig 10: prep-token direction vs assistant-token direction
def fig10_prep_direction():
    labs = ["hysterical", "angry", "desperate", "panicked", "depressed", "grief-stricken", "clinical_depression", "calm"]
    variants = [("assistant tokens, pooled", "spiral_direction_cosines.csv"), ("assistant tokens, within-turn", "spiral_direction_cosines_within.csv"),
                ("prep token, pooled", "spiral_direction_cosines_prep.csv"), ("prep token, within-turn", "spiral_direction_cosines_prep_within.csv")]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(labs)); w = 0.2
    for i, (name, f) in enumerate(variants):
        d = pd.read_csv(RESULTS_DIR / "analysis" / "gemma3_27b" / f); d = d[d.layer == 24]
        ax.barh(y + (i - 1.5) * w, [float(d[d.label == e].cosine.iloc[0]) for e in labs], height=w - 0.02, color=PAL[i], label=name)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("cosine(high-minus-low direction, story vector), layer 24")
    ax.set_title("Gemma 3 27B: the state before a bad turn points the same way as the state during it", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig10_prep_direction.png")


# ---------------------------------------------------------------- phase 4 helpers
def _cell_glob(model_key, pattern):
    """First existing steering cell matching results/spiral/<model>/extended_<pattern>; returns (tag, stats) or (None, None)."""
    hits = sorted((RESULTS_DIR / "spiral" / model_key).glob("extended_" + pattern))
    for h in hits:
        st = _cell_stats(model_key, h.name[len("extended_"):])
        if st:
            return h.name[len("extended_"):], st
    return None, None


def _coh(model_key, tag):
    from dprobe.steer import coherence
    return coherence(RESULTS_DIR / "spiral" / model_key / f"extended_{tag}" / "transcripts.jsonl")


def phase4_table(model_key, rows):
    """rows: list of (name, glob pattern). Prints a markdown table and returns {name: stats}."""
    out = {}
    print(f"| {model_key} cell | tag | n | mean | % ≥5 | turn-8 mean | anger | fear | depression | frustration | coherence |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, pat in rows:
        tag, st = _cell_glob(model_key, pat)
        if not st:
            print(f"| {name} | (not run) | | | | | | | | | |"); continue
        c = _coh(model_key, tag); out[name] = {**st, "tag": tag, **c}
        pet = " | ".join(f"{st[k]:.1f}" if k in st else "—" for k in ("anger", "fear", "depression", "frustration"))
        print(f"| {name} | {tag} | {st['n']} | {st['mean']:.2f} | {st['pct5']:.0f} | {st['t8']:.2f} | {pet} | {c['distinct_ratio']:.2f} / {c['repeated_3gram_share']:.2f} / {c['short_share']:.2f} |")
    return out


# ---------------------------------------------------------------- fig 11: Gemma 4 calm x axis factorial
def fig11_gemma4_factorial():
    calm_lv, axis_lv = [0, -2, -4], [0, -1, -2]
    cells = {(0, 0): "steer-depressed@34-44v+0", (0, -1): "steer-assistant_axis@34-44v-1", (0, -2): "steer-assistant_axis@34-44v-2",
             (-2, 0): "steer-calm@34-44v-2", (-4, 0): "steer-calm@34-44v-4", (-2, -2): "combo-calm-2_assistant_axis-2@34-44",
             (-4, -1): "combo-calm-4_assistant_axis-1@34-44"}
    M = np.full((3, 3), np.nan); N = np.zeros((3, 3), dtype=int); P5 = np.full((3, 3), np.nan)
    for (c, a), tag in cells.items():
        st = _cell_stats("gemma4_31b", tag)
        if st and st["n"] >= 8:
            i, j = calm_lv.index(c), axis_lv.index(a); M[i, j] = st["mean"]; N[i, j] = st["n"]; P5[i, j] = st["pct5"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cmap = matplotlib.colormaps["Oranges"].copy(); cmap.set_bad("#e6e5e1")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=10)
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                txt, col = ("invalid\n(empty outputs)" if (calm_lv[i], axis_lv[j]) in () else "not run"), INK2
            else:
                txt, col = f"{M[i, j]:.2f}\n{P5[i, j]:.0f}% ≥ 5\n(n={N[i, j]})", (INK if M[i, j] < 6 else "white")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{a:+d}× axis" if a else "axis 0" for a in axis_lv]); ax.set_yticks(range(3)); ax.set_yticklabels([f"{c:+d}× calm" if c else "calm 0" for c in calm_lv])
    ax.set_xlabel("assistant axis steering (− = away from the assistant)"); ax.set_ylabel("calm steering (− = less calm)"); ax.grid(False)
    ax.set_title("Gemma 4 31B: mean frustration score, calm × assistant axis\n(layers 34–44, 16 rollouts per cell; −8× calm was degenerate)", fontsize=9.5)
    _save(fig, "fig11_gemma4_factorial.png")


# ---------------------------------------------------------------- fig 12: Gemma 3 spiral family + axis, both bands
def fig12_gemma3_family_axis():
    inv = "#d6d5d0"
    rows_a = [("unsteered", "steer-depressed@34-46v+0", 0), ("+2 calm", "steer-calm@34-46v+2", 0), ("−2 calm", "steer-calm@34-46v-2", 0),
              ("−2 hysterical", "steer-hysterical@34-46v-2", 0), ("+2 hysterical", "steer-hysterical@34-46v+2", 1),
              ("−2 panicked", "steer-panicked@34-46v-2", 0), ("−1 panicked", "steer-panicked@34-46v-1", 0), ("+1 panicked", "steer-panicked@34-46v+1", 0), ("+2 panicked", "steer-panicked@34-46v+2", 0),
              ("+2 assistant axis", "steer-assistant_axis@34-46v+2", 0), ("+1 assistant axis", "steer-assistant_axis@34-46v+1", 0),
              ("−1 assistant axis", "steer-assistant_axis@34-46v-1", 0), ("−2 assistant axis", "steer-assistant_axis@34-46v-2", 1)]
    rows_b = [("unsteered", "steer-calm@20-26v+0", 0), ("+calm", "steer-calm@20-26v+[0-9]*", 0), ("−calm", "steer-calm@20-26v-[0-9]*", 0),
              ("+assistant axis", "steer-assistant_axis@20-26v+[0-9]*", 0), ("−assistant axis", "steer-assistant_axis@20-26v-[0-9]*", 0),
              ("+axis minus calm", "steer-assistant_axis_minus_calm@20-26v+[0-9]*", 0), ("−axis minus calm", "steer-assistant_axis_minus_calm@20-26v-[0-9]*", 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, rows, title in zip(axes, [rows_a, rows_b], ["Layers 34–46, fixed multipliers", "Layers 20–26 (entangled band), calibrated"]):
        names, vals, cols, invalid = [], [], [], []
        for name, pat, bad in rows:
            tag, st = _cell_glob("gemma3_27b", pat)
            mult = tag.split("v")[-1] if tag and "@20-26" in tag and "v+0" not in tag else ""
            names.append(f"{name} ({mult}×)" if mult else name)
            vals.append(st["mean"] if st else np.nan); invalid.append(bad)
            cols.append(inv if bad else (PAL[6] if name == "unsteered" else (PAL[2] if name.startswith("+") else PAL[1])))
        y = np.arange(len(names)); ax.barh(y, vals, color=cols, height=0.6)
        for yi, v, bad in zip(y, vals, invalid):
            ax.text((v if not np.isnan(v) else 0) + 0.1, yi, ("not run" if np.isnan(v) else f"{v:.2f}" + (" (invalid: incoherent)" if bad else "")), va="center", fontsize=8, color=INK2)
        if not np.isnan(vals[0]):
            ax.axvline(vals[0], color=INK2, lw=0.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlim(0, 10); ax.set_xlabel("mean frustration judge score, all turns"); ax.set_title(title)
    fig.suptitle("Gemma 3 27B: the spiral family and the assistant axis are causal in both directions (green = +, orange = −, grey = degenerate; 16 rollouts per cell)", fontsize=10, color=INK2)
    _save(fig, "fig12_gemma3_family_axis.png")


if __name__ == "__main__":
    for f in (fig1_behaviour, fig2_spiral_direction, fig3_prediction, fig4_prep_curves, fig5_geometry, fig6_steering_gemma3, fig7_axis, fig8_steering_gemma4, fig9_prefill,
              fig10_prep_direction, fig11_gemma4_factorial, fig12_gemma3_family_axis):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"!! {f.__name__} failed: {type(e).__name__}: {e}")
