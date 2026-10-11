"""Multi-turn prefill: Gemma 3's high-frustration turns 1-4 as each model's own history, the model writes turns 5-8.

Top: probes just before each reply (z vs neutral stories, two-thirds depth); turns 1-4 are Gemma 3's text.
Bottom: judge frustration score by turn (solid) against the same model's unprefilled run (dashed).
Run: DPROBE_RESULTS=<main repo>/results uv run python scripts/make_prefill_mt_figure.py
Output: results/analysis/organisms/fig_prefill4_by_turn.png and prefill4_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import analyze_organisms as A  # noqa: E402

from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, INK2 = A.plt, A.INK2
RNG = np.random.default_rng(0)
# (key, layer, name, tag of the unprefilled baseline run)
COLS = [("gemma3_27b", 40, "Gemma 3", "local"), ("gemma3_27b_dpo", 40, "Gemma 3 + DPO", "local"),
        ("gemma3_27b_bct", 40, "Gemma 3 + BCT", "local"), ("gemma4_31b", 39, "Gemma 4", "")]
TAG = "prefill4"
K = 4


def judge_by_turn(mk, tag):
    J = load_judgments(transcripts_path(mk, "extended", tag))
    by = {}
    for (_, k), r in J.items():
        by.setdefault(k, []).append(r["rating"])
    ks = np.array(sorted(by))
    m = np.array([np.mean(by[k]) for k in ks])
    ci = np.array([np.percentile(RNG.choice(by[k], (2000, len(by[k]))).mean(1), [2.5, 97.5]) for k in ks])
    ge5 = np.array([100 * np.mean(np.array(by[k]) >= 5) for k in ks])
    return ks + 1, m, ci, ge5


def main():
    fig, axes = plt.subplots(2, len(COLS), figsize=(4.4 * len(COLS), 6.6), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    rows = []
    for col, (mk, L, name, base_tag) in enumerate(COLS):
        try:
            cur = A.probe_curves(mk, TAG, layer=L)
            t, m, ci, ge5 = judge_by_turn(mk, TAG)
        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
            print(f"skip {name}: {e}"); continue
        ax = axes[0, col]; ends = []
        ax.axvspan(0.5, K + 0.5, color=INK2, alpha=0.08, lw=0)
        for e, c in A.PROBES:
            v = cur[e]
            ax.plot(np.arange(1, len(v) + 1), v, color=c, lw=2, marker="o", ms=3, label=e)
            ends.append((len(v), float(v[-1]), e))
        ax.set_ylim(-5.2, 7.2); A.mf._end_labels(ax, ends)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title(name)
        ax.text(K / 2 + 0.5, 6.7, "Gemma 3's turns", ha="center", fontsize=8, color=INK2)
        ax = axes[1, col]
        ax.axvspan(0.5, K + 0.5, color=INK2, alpha=0.08, lw=0)
        ax.fill_between(t, ci[:, 0], ci[:, 1], color="#3d3c38", alpha=0.15, lw=0)
        ax.plot(t, m, color="#3d3c38", lw=2, marker="o", ms=3, label="prefilled")
        bt, bm, _, _ = judge_by_turn(mk, base_tag)
        ax.plot(bt, bm, color="#3d3c38", lw=1.2, ls="--", label="own run, no prefill")
        ax.axhline(5, color=INK2, lw=0.6, ls=":"); ax.set_ylim(0, 10); ax.set_xlabel("turn")
        ax.set_xticks(range(1, 9)); ax.set_xlim(0.5, 9.6)
        r = {"model": name, "conversations": len({c for c, _ in load_judgments(transcripts_path(mk, "extended", TAG))}),
             "judge turn 4 (prefill)": m[K - 1], "judge turn 5": m[K], "judge turn 8": m[-1], "% >=5 turns 5-8": ge5[K:].mean(),
             "own run judge turn 5": bm[K], "own run judge turn 8": bm[-1]}
        for e, _ in A.PROBES:
            r[f"{e} turn 4"], r[f"{e} turn 8"] = float(cur[e][K - 1]), float(cur[e][-1])
        rows.append(r)
    axes[0, 0].set_ylabel("probe before the reply (z)"); axes[1, 0].set_ylabel("frustration score (0–10)")
    axes[1, 0].legend(fontsize=7.5, loc="upper left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(A.OUT / "fig_prefill4_by_turn.png", dpi=150); plt.close(fig)
    df = pd.DataFrame(rows).set_index("model")
    df.round(2).to_csv(A.OUT / "prefill4_summary.csv")
    print(df.round(1).T.to_string())
    print("wrote", A.OUT / "fig_prefill4_by_turn.png")


if __name__ == "__main__":
    main()
