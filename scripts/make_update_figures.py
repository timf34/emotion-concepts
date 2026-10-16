"""Figures for the Oct 2026 research update (Finding 8).

fig10_entanglement_artefact.png  the persona-emotion cosine per model, as first measured vs with every coordinate
                                 standardised, plus the scale-free calm-end test
fig11_causal_2733.png            rejection-eval steering: coordinate 2733 removed / alone / clamped, and single SAE
                                 features of the assistant axis
Run: uv run python scripts/make_update_figures.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

from dprobe.config import RESULTS_DIR  # noqa: E402
from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, PAL, INK, INK2 = mf.plt, mf.PAL, mf.INK, mf.INK2
OUT = RESULTS_DIR / "analysis" / "figures" / "writeup"
RNG = np.random.default_rng(0)


def fig10():
    df = pd.read_csv(RESULTS_DIR / "analysis" / "crossfamily" / "robust.csv").set_index("model")
    order = ["Gemma 3 27B", "Gemma 3 base", "Gemma 2 27B", "gpt-oss-20b", "Gemma 4 31B", "Qwen3 32B", "GLM-4.5-Air",
             "Mistral Small 3 24B", "Llama 3.3 70B", "OLMo 2 32B"]
    order = [m for m in order if m in df.index]
    d = df.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(order)); w = 0.38
    axes[0].barh(y - w / 2, d["cos"], height=w - 0.03, color=PAL[0], label="as first measured (cosine)")
    axes[0].barh(y + w / 2, d["zscored"], height=w - 0.03, color=PAL[3], label="every coordinate standardised")
    axes[0].set_yticks(y); axes[0].set_yticklabels(order); axes[0].invert_yaxis(); axes[0].set_xlim(0, 0.3)
    axes[0].set_xlabel("|cosine| of assistant axis with emotion vectors (median, mid-depth)")
    axes[0].set_title("The 'entanglement' disappears without one coordinate"); axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(axis="y", visible=False)
    for yi, (c, z) in enumerate(zip(d["cos"], d["zscored"])):
        axes[0].text(c + 0.004, yi - w / 2, f"{c:.2f}", va="center", fontsize=7.5, color=INK2)
    axes[1].barh(y, 100 * d["calm_end"], height=0.6, color=[PAL[0] if "Gemma 3" in m else INK2 for m in order])
    axes[1].axvline(50, color=INK2, lw=0.8, ls="--")
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].invert_yaxis(); axes[1].set_xlim(30, 85)
    axes[1].set_xlabel("% of story pairs where the calm story sits\nnearer the assistant end (50% = no relation)")
    axes[1].set_title("Scale-free test: Gemma 3 is ordinary"); axes[1].grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "fig10_entanglement_artefact.png", dpi=150); plt.close(fig)
    print("wrote fig10")


def cell(tag):
    J = load_judgments(transcripts_path("gemma3_27b", "extended", tag))
    c = defaultdict(list)
    for (cid, _), r in J.items():
        c[cid].append(r["rating"])
    m = np.array([np.mean(v) for v in c.values()])
    bs = RNG.choice(m, (4000, len(m))).mean(1)
    return m.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(m)


def bars(ax, rows, title, base):
    y = np.arange(len(rows))
    for i, (name, tag, col) in enumerate(rows):
        try:
            m, lo, hi, n = cell(tag)
        except (FileNotFoundError, ValueError):
            continue
        ax.barh(i, m, height=0.6, color=col)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.text(hi + 0.12, i, f"{m:.1f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(base, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 8)
    ax.set_xlabel("mean frustration score (0–10), 16 conversations each, 95% interval"); ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", visible=False)


def fig11():
    G, B, R, C = INK2, PAL[0], PAL[1], PAL[2]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.8), gridspec_kw={"height_ratios": [5, 7]})
    base = cell("steer-depressed@34-46v+0")[0]
    bars(axes[0], [("unsteered", "steer-depressed@34-46v+0", G),
                   ("push off the assistant (axis −2×, L20–26)", "steer-assistant_axis@20-26v-2", B),
                   ("same push, coordinate 2733 removed", "axis-no2733@20-26v-2", B),
                   ("only the push's coordinate-2733 part", "axis-only2733@20-26v-2", R),
                   ("2733 clamped at the assistant's level", "clamp2733@16-26", R)],
         "Coordinate 2733 plays no part in the spiral", base)
    bars(axes[1], [("unsteered", "steer-depressed@34-46v+0", G),
                   ("push off the assistant (axis −1×, L34–46)", "axisfeat-control@34-46", B),
                   ("'vulgar or aggressive speech' feature", "axisfeat-f998@34-46", C),
                   ("'ancient prophecies and quests' feature", "axisfeat-f1101@34-46", C),
                   ("'initiating interactions' feature", "axisfeat-f47828@34-46", INK2),
                   ("'I am' feature", "axisfeat-f732@34-46", INK2),
                   ("random direction, same strength", "axisfeat-random@34-46", G)],
         "Pushing along single SAE features of the axis (same strength)", base)
    fig.tight_layout(); fig.savefig(OUT / "fig11_causal_2733.png", dpi=150); plt.close(fig)
    print("wrote fig11")


if __name__ == "__main__":
    fig10(); fig11()
