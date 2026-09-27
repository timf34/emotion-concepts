"""Figures for results/analysis/WRITEUP.md (the short, per-experiment write-up).

Short titles, no suptitles (details live in the captions in WRITEUP.md), 95% bootstrap intervals on every steering bar.
Reuses the style and helpers of scripts/make_figures.py. Output: results/analysis/figures/writeup/*.png
Run: uv run python scripts/make_writeup_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402  (style, palette, helpers)

from dprobe.analysis import _load, _score_matrix  # noqa: E402
from dprobe.config import RESULTS_DIR  # noqa: E402
from dprobe.extract import load_vectors, vectors_dir  # noqa: E402
from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, PAL, INK, INK2, GRID = mf.plt, mf.PAL, mf.INK, mf.INK2, mf.GRID
OUT = RESULTS_DIR / "analysis" / "figures" / "writeup"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
G3, G4 = PAL[0], PAL[1]           # model colours, fixed everywhere


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def cell(model_key, tag, B=4000):
    """Mean judge score over all turns with a per-conversation bootstrap 95% CI, plus the turn-1 mean."""
    J = load_judgments(transcripts_path(model_key, "extended", tag))
    if not J:
        return None
    by, t1 = {}, []
    for (cid, t), r in J.items():
        by.setdefault(cid, []).append(r["rating"])
        if t == 0:
            t1.append(r["rating"])
    m = np.array([np.mean(v) for v in by.values()])
    bs = RNG.choice(m, (B, len(m))).mean(1)
    return {"mean": m.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "t1": float(np.mean(t1)), "n": len(m)}


def bars(ax, rows, model_key, base=None, show_t1=True):
    """Horizontal bars with CIs. rows: (label, tag, colour). Returns stats."""
    st = [cell(model_key, tag) for _, tag, _ in rows]
    y = np.arange(len(rows))
    for yi, (lab, _, col), s in zip(y, rows, st):
        if s is None:
            ax.text(0.1, yi, "not run", va="center", fontsize=8, color=INK2)
            continue
        ax.barh(yi, s["mean"], color=col, height=0.62)
        ax.errorbar(s["mean"], yi, xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
        ax.text(s["hi"] + 0.15, yi, f"{s['mean']:.1f}", va="center", fontsize=8, color=INK2)
        if show_t1:
            ax.plot(s["t1"], yi, marker="o", ms=5, mfc="white", mec=INK, mew=1, zorder=5)
    if base is not None:
        ax.axvline(base, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 10.5)
    ax.set_xlabel("frustration score (0–10)")
    ax.grid(axis="y", visible=False)
    return st


def t1_legend(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [plt.Rectangle((0, 0), 1, 1, color="#9a9890"), Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5)]
    ax.legend(h, ["mean over all 8 turns (95% CI)", "turn 1, before any rejection"], fontsize=7.5, loc=loc)


# ------------------------------------------------------------------ 1. replication
def fig1():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for mk, name, col in [("gemma3_27b", "Gemma 3 27B", G3), ("gemma4_31b", "Gemma 4 31B", G4)]:
        J = load_judgments(transcripts_path(mk)); by = {}
        for (cid, t), r in J.items():
            by.setdefault(t, []).append(r["rating"])
        t = np.array(sorted(by)); m = np.array([np.mean(by[k]) for k in t])
        ci = np.array([np.percentile(RNG.choice(by[k], (2000, len(by[k]))).mean(1), [2.5, 97.5]) for k in t])
        ax.fill_between(t + 1, ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
        ax.plot(t + 1, m, color=col, lw=2, marker="o", ms=4, label=name)
        ax.annotate(name, (t[-1] + 1, m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_title("Frustration score by turn"); ax.set_xlabel("turn"); ax.set_ylabel("frustration score (0–10)")
    ax.set_xticks(range(1, 9)); ax.set_xlim(0.7, 10.2); ax.set_ylim(0, 8); ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "fig1_replication.png")


# ------------------------------------------------------------------ 2. what tracks spiral intensity
TRACK = ["tormented", "hysterical", "desperate", "overwhelmed", "panicked", "anxiety_panic",
         "depressed", "miserable", "sad", "clinical_depression", "frustrated", "calm", "hopeful"]


def within_turn_rho(mk, L, turns=range(2, 8)):
    P, J, _ = _load(mk); S = _score_matrix(P, J)
    proj = P["proj_mean"][:, :, P["layers"].index(L)].float().numpy()
    labs = P["labels"]
    return {e: float(np.nanmean([spearmanr(proj[:, k, labs.index(e)], S[:, k], nan_policy="omit")[0] for k in turns])) for e in TRACK}


def fig2():
    r24, r40 = within_turn_rho("gemma3_27b", 24), within_turn_rho("gemma3_27b", 40)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(TRACK)); w = 0.38
    ax.barh(y - w / 2, [r24[e] for e in TRACK], height=w - 0.03, color=PAL[0], label="layer 24")
    ax.barh(y + w / 2, [r40[e] for e in TRACK], height=w - 0.03, color=PAL[2], label="layer 40 (two-thirds)")
    ax.set_yticks(y); ax.set_yticklabels(TRACK); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-0.45, 0.6); ax.set_xlabel("within-turn Spearman ρ with the frustration score")
    ax.set_title("What tracks the spiral (Gemma 3)"); ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, "fig2_what_tracks.png")
    return r24, r40


# ------------------------------------------------------------------ 3. steering Gemma 3
def fig3():
    B = "#9a9890"
    rows = [("unsteered", "steer-depressed@34-46v+0", B),
            ("+2 calm", "steer-calm@34-46v+2", PAL[2]), ("−2 calm", "steer-calm@34-46v-2", PAL[2]),
            ("+2 hysterical", "steer-hysterical@34-46v+2", PAL[1]), ("−2 hysterical", "steer-hysterical@34-46v-2", PAL[1]),
            ("−2 panicked", "steer-panicked@34-46v-2", PAL[1]), ("−1 panicked", "steer-panicked@34-46v-1", PAL[1]), ("+1 panicked", "steer-panicked@34-46v+1", PAL[1]),
            ("+2 clinical depression", "steer-clinical_depression@34-46v+2", PAL[6]), ("−2 clinical depression", "steer-clinical_depression@34-46v-2", PAL[6]),
            ("+2 depressed", "steer-depressed@34-46v+2", PAL[4]), ("−2 depressed", "steer-depressed@34-46v-2", PAL[4])]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    st = bars(ax, rows, "gemma3_27b", base=cell("gemma3_27b", rows[0][1])["mean"])
    ax.set_title("Steering Gemma 3"); t1_legend(ax)
    save(fig, "fig3_steer_gemma3.png")
    return dict(zip([r[0] for r in rows], st))


# ------------------------------------------------------------------ 4. internal state vs what the model says, by turn
def fig4():
    labs = [("panicked", PAL[1]), ("desperate", PAL[0]), ("frustrated", PAL[3]), ("calm", PAL[2]), ("depressed", PAL[6])]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    for col, (mk, f, name, mcol) in enumerate([("gemma3_27b", "turn_curves_L40.csv", "Gemma 3", G3), ("gemma4_31b", "turn_curves_L39.csv", "Gemma 4", G4)]):
        ax = axes[0, col]; t = mf.pd.read_csv(RESULTS_DIR / "analysis" / mk / f); ends = []
        for e, c in labs:
            ax.plot(t.turn, t[f"z_prep::{e}"], color=c, lw=2, marker="o", ms=3, label=e)
            ends.append((t.turn.iloc[-1], t[f"z_prep::{e}"].iloc[-1], e))
        ax.set_ylim(-4.8, 6.8); mf._end_labels(ax, ends)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title(name)
        # judge score by turn, same conversations
        J = load_judgments(transcripts_path(mk)); by = {}
        for (cid, k), r in J.items():
            by.setdefault(k, []).append(r["rating"])
        ks = np.array(sorted(by)); m = np.array([np.mean(by[k]) for k in ks])
        ci = np.array([np.percentile(RNG.choice(by[k], (2000, len(by[k]))).mean(1), [2.5, 97.5]) for k in ks])
        ax = axes[1, col]
        ax.fill_between(ks + 1, ci[:, 0], ci[:, 1], color="#3d3c38", alpha=0.15, lw=0)
        ax.plot(ks + 1, m, color="#3d3c38", lw=2, marker="o", ms=3)
        ax.axhline(5, color=INK2, lw=0.6, ls=":"); ax.set_ylim(0, 10); ax.set_xlabel("turn")
        ax.set_xticks(range(1, 9)); ax.set_xlim(0.7, 10)
    axes[0, 0].set_ylabel("probe before the reply (z)"); axes[1, 0].set_ylabel("frustration score (0–10)")
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    axes[1, 0].text(1.1, 5.25, "breakdown threshold", fontsize=7.5, color=INK2)
    save(fig, "fig4_state_by_turn.png", rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ probe change from turn 1 to turn 8, both models
CHANGE = ["hysterical", "desperate", "panicked", "frustrated", "tormented", "depressed", "sad", "clinical_depression", "calm"]


def fig_change():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y = np.arange(len(CHANGE)); w = 0.38; out = {}
    for i, (mk, f, name, col) in enumerate([("gemma3_27b", "turn_curves_L40.csv", "Gemma 3", G3), ("gemma4_31b", "turn_curves_L39.csv", "Gemma 4", G4)]):
        t = mf.pd.read_csv(RESULTS_DIR / "analysis" / mk / f)
        d = [float(t[f"z_prep::{e}"].iloc[-1] - t[f"z_prep::{e}"].iloc[0]) for e in CHANGE]; out[name] = dict(zip(CHANGE, d))
        ax.barh(y + (i - 0.5) * w, d, height=w - 0.03, color=col, label=name)
    ax.set_yticks(y); ax.set_yticklabels([e.replace("_", " ") for e in CHANGE]); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("change in probe before the reply, turn 1 to turn 8 (z)"); ax.set_title("What rises under rejection")
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, "figA_probe_change.png")
    return out


# ------------------------------------------------------------------ before the first vs the last reply (arrows)
ARROWS = ["hysterical", "desperate", "panicked", "frustrated", "depressed", "sad", "clinical_depression", "calm"]


def fig_arrows():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    up, down = PAL[1], PAL[0]
    for ax, (mk, f, name) in zip(axes, [("gemma3_27b", "turn_curves_L40.csv", "Gemma 3"), ("gemma4_31b", "turn_curves_L39.csv", "Gemma 4")]):
        t = mf.pd.read_csv(RESULTS_DIR / "analysis" / mk / f)
        for i, e in enumerate(ARROWS):
            a, b = float(t[f"z_prep::{e}"].iloc[0]), float(t[f"z_prep::{e}"].iloc[-1]); col = up if b > a else down
            ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2, shrinkA=4, shrinkB=4, mutation_scale=12))
            ax.plot(a, i, "o", ms=7, mfc="white", mec=col, mew=1.6, zorder=4)
            ax.plot(b, i, "o", ms=7, color=col, zorder=4)
        ax.axvline(0, color=INK2, lw=0.6); ax.set_title(name); ax.set_xlim(-5, 7)
        ax.set_xlabel("probe before the reply (z)"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ARROWS))); axes[0].set_yticklabels([e.replace("_", " ") for e in ARROWS]); axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=7, mew=1.4), Line2D([], [], marker="o", ls="", color=INK2, ms=7),
         Line2D([], [], color=up, lw=2), Line2D([], [], color=down, lw=2)]
    fig.legend(h, ["turn 1", "turn 8", "rises", "falls"], loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    save(fig, "figB_before_after.png", rect=(0, 0, 1, 0.93))


# ------------------------------------------------------------------ 5. assistant axis vs emotion vectors
def axis_emotion_cos(mk):
    """{emotion: {layer: cosine(assistant axis, emotion vector)}} over the analysis layers."""
    ax_ = torch.load(vectors_dir(mk) / "vectors_external_dn.pt")["assistant_axis"]; emo = load_vectors(mk, "emotions", True)
    return {e: {l: float(ax_[l] @ emo[e][l] / (ax_[l].norm() * emo[e][l].norm())) for l in sorted(ax_)} for e in emo}


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for mk, name, col in [("gemma3_27b", "Gemma 3", G3), ("gemma4_31b", "Gemma 4", G4)]:
        C = axis_emotion_cos(mk); ls = sorted(next(iter(C.values())))
        A = np.abs(np.array([[C[e][l] for l in ls] for e in C]))          # [42, L]
        axes[0].fill_between(ls, np.percentile(A, 10, 0), np.percentile(A, 90, 0), color=col, alpha=0.15, lw=0)
        axes[0].plot(ls, np.median(A, 0), color=col, lw=2, label=name)
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("|cosine| with the assistant axis")
    axes[0].set_title("Assistant axis vs all 42 emotions"); axes[0].legend(fontsize=8, loc="upper right"); axes[0].set_ylim(0, 0.6)
    labs = ["hysterical", "angry", "desperate", "panicked", "frustrated", "depressed", "sad", "calm", "hopeful"]
    y = np.arange(len(labs)); w = 0.36
    for i, (mk, name, col) in enumerate([("gemma3_27b", "Gemma 3", G3), ("gemma4_31b", "Gemma 4", G4)]):
        base, _, dv = mf._axis(mk); L = 24
        R = torch.stack([torch.load(f).float()[L] for f in sorted(base.glob("role_vectors/*.pt"))])
        C = torch.load(vectors_dir(mk) / "neutral_pca.pt")[L]["components"]
        D = R - dv[L]; D = D - (D @ C.T) @ C
        emo = load_vectors(mk, "emotions", True); V = torch.stack([emo[e][L] for e in labs]); Vn = V / V.norm(dim=1, keepdim=True)
        cos = ((D / D.norm(dim=1, keepdim=True)) @ Vn.T).mean(0).numpy()
        axes[1].barh(y + (i - 0.5) * w, cos, height=w - 0.03, color=col, label=name)
    axes[1].set_yticks(y); axes[1].set_yticklabels(labs); axes[1].invert_yaxis(); axes[1].axvline(0, color=INK2, lw=0.6)
    axes[1].set_xlabel("mean cosine, 275 roles, layer 24"); axes[1].set_title("Role personas' affect"); axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(axis="y", visible=False)
    save(fig, "fig5_axis_vs_emotion.png")


def fig5b():
    CG3, CG4 = axis_emotion_cos("gemma3_27b"), axis_emotion_cos("gemma4_31b")
    early = [l for l in next(iter(CG3.values())) if 6 <= l <= 26]
    order = sorted(CG3, key=lambda e: -np.mean([CG3[e][l] for l in early]))      # most assistant-aligned first (in Gemma 3)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#1d5aa6", "#8fb5e0", "#e6e5e1", "#f0a37a", "#c2410c"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 9.5), sharey=True)
    for ax, (C, name) in zip(axes, [(CG3, "Gemma 3"), (CG4, "Gemma 4")]):
        ls = sorted(next(iter(C.values()))); M = np.array([[C[e][l] for l in ls] for e in order])
        im = ax.imshow(M, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(0, len(ls), 3)); ax.set_xticklabels([ls[i] for i in range(0, len(ls), 3)], fontsize=8)
        ax.set_xlabel("layer"); ax.set_title(name); ax.grid(False)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02); cb.set_label("cosine with the assistant axis")
    fig.savefig(OUT / "fig5b_axis_all_emotions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "fig5b_axis_all_emotions.png")


# ------------------------------------------------------------------ 6. steering Gemma 3's axis
def fig6():
    B, AX, RS = "#9a9890", PAL[0], PAL[3]
    a = [("unsteered", "steer-depressed@34-46v+0", B), ("+2 axis", "steer-assistant_axis@34-46v+2", AX), ("+1 axis", "steer-assistant_axis@34-46v+1", AX),
         ("−1 axis", "steer-assistant_axis@34-46v-1", AX)]
    b = [("unsteered", "steer-calm@20-26v+0", B), ("+2 axis", "steer-assistant_axis@20-26v+2", AX), ("−2 axis", "steer-assistant_axis@20-26v-2", AX),
         ("+2 axis minus calm", "steer-assistant_axis_minus_calm@20-26v+2", RS), ("−2 axis minus calm", "steer-assistant_axis_minus_calm@20-26v-2", RS),
         ("−4 axis minus calm", "steer-assistant_axis_minus_calm@20-26v-4", RS), ("−8 axis minus calm", "steer-assistant_axis_minus_calm@20-26v-8", RS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [0.8, 1]})
    sa = bars(axes[0], a, "gemma3_27b", base=cell("gemma3_27b", a[0][1])["mean"]); axes[0].set_title("Layers 34–46")
    sb = bars(axes[1], b, "gemma3_27b", base=cell("gemma3_27b", b[0][1])["mean"]); axes[1].set_title("Layers 20–26")
    t1_legend(axes[1])
    save(fig, "fig6_steer_axis_gemma3.png")
    return dict(zip([r[0] for r in a], sa)), dict(zip([r[0] for r in b], sb))


# ------------------------------------------------------------------ 7. steering Gemma 4
def fig7():
    B = "#9a9890"
    rows = [("unsteered", "steer-depressed@34-44v+0", B),
            ("+1 depressed", "steer-depressed@34-44v+1", PAL[4]), ("+4 clinical depression", "steer-clinical_depression@34-44v+4", PAL[6]),
            ("+2 hysterical", "steer-hysterical@34-44v+2", PAL[1]), ("+4 desperate", "steer-desperate@34-44v+4", PAL[1]), ("+4 panicked", "steer-panicked@34-44v+4", PAL[1]),
            ("−2 axis", "steer-assistant_axis@34-44v-2", PAL[0]), ("−2 calm", "steer-calm@34-44v-2", PAL[2]), ("−4 calm", "steer-calm@34-44v-4", PAL[2])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    st = bars(axes[0], rows, "gemma4_31b", base=None); axes[0].set_title("One vector at a time"); t1_legend(axes[0], loc="center right")
    # calm x axis grid
    calm_lv, axis_lv = [0, -2, -4], [0, -1, -2]
    tags = {(0, 0): "steer-depressed@34-44v+0", (0, -1): "steer-assistant_axis@34-44v-1", (0, -2): "steer-assistant_axis@34-44v-2",
            (-2, 0): "steer-calm@34-44v-2", (-4, 0): "steer-calm@34-44v-4", (-2, -2): "combo-calm-2_assistant_axis-2@34-44", (-4, -1): "combo-calm-4_assistant_axis-1@34-44"}
    M = np.full((3, 3), np.nan); T1 = np.full((3, 3), np.nan)
    for (c, a), tag in tags.items():
        s = cell("gemma4_31b", tag); M[calm_lv.index(c), axis_lv.index(a)] = s["mean"]; T1[calm_lv.index(c), axis_lv.index(a)] = s["t1"]
    cmap = mf.matplotlib.colormaps["Oranges"].copy(); cmap.set_bad(GRID)
    ax = axes[1]; ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=10)
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                ax.text(j, i, "not run", ha="center", va="center", fontsize=8.5, color=INK2)
            else:
                ax.text(j, i, f"{M[i, j]:.1f}\n(turn 1: {T1[i, j]:.1f})", ha="center", va="center", fontsize=9, color=INK if M[i, j] < 6 else "white")
    ax.set_xticks(range(3)); ax.set_xticklabels(["axis 0", "−1× axis", "−2× axis"]); ax.set_yticks(range(3)); ax.set_yticklabels(["calm 0", "−2× calm", "−4× calm"])
    ax.set_title("Calm × assistant axis"); ax.grid(False)
    save(fig, "fig7_steer_gemma4.png")
    return dict(zip([r[0] for r in rows], st)), M, T1


# ------------------------------------------------------------------ 8. prefill recovery
def fig8():
    runs = [("Gemma 3 continues its own spiral", "gemma3_27b", "from-gemma3_27b_t6_unsteered", 40, G3),
            ("Gemma 4 continues it", "gemma4_31b", "from-gemma3_27b_t6_unsteered", 39, G4),
            ("Gemma 4, −4× assistant axis", "gemma4_31b", "from-gemma3_27b_t6_steer-assistant_axis-4", 39, PAL[2])]
    labs = [("assistant_axis", "Assistant axis"), ("calm", "Calm"), ("desperate", "Desperate")]
    buckets = [0, 32, 64, 128, 192, 256, 384, 512]; xs = [(buckets[b] + buckets[b + 1]) / 2 for b in range(len(buckets) - 1)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for name, mk, tag, L, col in runs:
        D = torch.load(RESULTS_DIR / "prefill" / mk / tag / "curves.pt"); li = D["layers"].index(L)
        Z = (D["cont"][:, :, li] - D["prefix_pool_mean"][li]) / D["prefix_pool_std"][li]
        for ax, (e, _) in zip(axes, labs):
            ei = D["labels"].index(e)
            ax.plot(xs, [float(np.nanmean(Z[:, buckets[b]:buckets[b + 1], ei].numpy())) for b in range(len(xs))], color=col, lw=2, marker="o", ms=3, label=name)
    for ax, (_, t) in zip(axes, labs):
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title(t); ax.set_xlabel("token of the reply"); ax.set_xticks([0, 128, 256, 384, 512])
    axes[0].set_ylabel("probe z vs the prefix")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5)
    save(fig, "fig8_prefill.png", rect=(0, 0.08, 1, 1))


if __name__ == "__main__":
    fig1(); r = fig2(); print("within-turn rho L24/L40:", {k: (round(r[0][k], 2), round(r[1][k], 2)) for k in TRACK})
    s3 = fig3(); print({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2), round(v["t1"], 2)) for k, v in s3.items()})
    fig4(); fig5(); fig6()
    s7, M, T1 = fig7(); print({k: (round(v["mean"], 2), round(v["t1"], 2)) for k, v in s7.items()}); print(M.round(2)); print(T1.round(2))
    fig8()
