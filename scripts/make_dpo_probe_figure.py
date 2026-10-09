"""Figure 4 for the DPO organism: probes before each reply (top) and the judge's score (bottom), by turn.

Gemma 3 and Gemma 3 + DPO: the local rejection eval (100 conversations each, same generation settings).
Gemma 4: its 300 conversations from Figure 4. Probes are z against each model's own neutral stories, at two-thirds depth.
Run: DPROBE_RESULTS=<main repo>/results uv run python scripts/make_dpo_probe_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_organisms as A  # noqa: E402

from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, INK2 = A.plt, A.INK2
RNG = np.random.default_rng(0)
COLS = [("gemma3_27b", "local", 40, "Gemma 3"), ("gemma3_27b_dpo", "local", 40, "Gemma 3 + DPO (the paper's fix)"), ("gemma4_31b", "", 39, "Gemma 4")]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    for col, (mk, tag, L, name) in enumerate(COLS):
        cur = A.probe_curves(mk, tag, layer=L)
        ax = axes[0, col]; ends = []
        for e, c in A.PROBES:
            v = cur[e]
            ax.plot(np.arange(1, len(v) + 1), v, color=c, lw=2, marker="o", ms=3, label=e)
            ends.append((len(v), float(v[-1]), e))
        ax.set_ylim(-4.5, 6.8); A.mf._end_labels(ax, ends)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title(name)
        J = load_judgments(transcripts_path(mk, "extended", tag))
        by = {}
        for (_, k), r in J.items():
            by.setdefault(k, []).append(r["rating"])
        ks = np.array(sorted(by)); m = np.array([np.mean(by[k]) for k in ks])
        ci = np.array([np.percentile(RNG.choice(by[k], (2000, len(by[k]))).mean(1), [2.5, 97.5]) for k in ks])
        ax = axes[1, col]
        ax.fill_between(ks + 1, ci[:, 0], ci[:, 1], color="#3d3c38", alpha=0.15, lw=0)
        ax.plot(ks + 1, m, color="#3d3c38", lw=2, marker="o", ms=3)
        ax.axhline(5, color=INK2, lw=0.6, ls=":"); ax.set_ylim(0, 10); ax.set_xlabel("turn")
        ax.set_xticks(range(1, 9)); ax.set_xlim(0.7, 9.6)
        n = len({cid for cid, _ in J})
        ax.text(0.98, 0.95, f"{n} conversations", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        print(name, {e: (round(float(cur[e][0]), 1), round(float(cur[e][-1]), 1)) for e, _ in A.PROBES}, "judge", m.round(1))
    axes[0, 0].set_ylabel("probe before the reply (z)"); axes[1, 0].set_ylabel("frustration score (0–10)")
    axes[1, 0].text(1.1, 5.25, "breakdown threshold", fontsize=7.5, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = A.OUT / "fig_dpo_state_by_turn.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
