"""Frustration by turn for three Gemma generations under the same rejection eval (fig 1 style).

Gemma 3 27B and Gemma 4 31B: the main 300-conversation runs (OpenRouter, 2048 tokens, Sonnet 5 judge).
Gemma 2 27B: the cross-family sweep (52 conversations, same 8-turn condition, rubric and judge; puzzle i mod 11;
OpenRouter's only Gemma 2 endpoint is int4-quantised).
Output: results/analysis/figures/writeup/fig1b_gemma_generations.png
Run: uv run python scripts/make_gemma_generations_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402

from dprobe.config import RESULTS_DIR  # noqa: E402
from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, PAL, INK2 = mf.plt, mf.PAL, mf.INK2
OUT = RESULTS_DIR / "analysis" / "figures" / "writeup"
RNG = np.random.default_rng(0)


def main_run(mk):
    """[conversations, turns] ratings from a main run."""
    J = load_judgments(transcripts_path(mk))
    conv = {}
    for (cid, k), r in J.items():
        conv.setdefault(cid, {})[k] = r["rating"]
    K = max(max(v) for v in conv.values()) + 1
    return np.array([[v.get(k, np.nan) for k in range(K)] for v in conv.values()], dtype=float)


def sweep_run(key):
    rows = [json.loads(l) for l in open(RESULTS_DIR / "sweep" / key / "conversations.jsonl")]
    return np.array([[t.get("rating", np.nan) if t.get("rating") is not None else np.nan for t in r["turns"]] for r in rows], dtype=float)


def main():
    data = [("Gemma 2 27B (sweep, n=52)", "#8a7a5c", sweep_run("gemma-2-27b")),
            ("Gemma 3 27B (n=300)", PAL[0], main_run("gemma3_27b")),
            ("Gemma 4 31B (n=300)", PAL[1], main_run("gemma4_31b"))]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    turns = np.arange(1, 9)
    for name, col, S in data:
        S = S[:, :8]
        m = np.nanmean(S, 0)
        bs = np.array([np.nanmean(S[RNG.integers(0, len(S), len(S))], 0) for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        axes[0].fill_between(turns, lo, hi, color=col, alpha=0.15, lw=0)
        axes[0].plot(turns, m, color=col, lw=2, marker="o", ms=3, label=name)
        pct = 100 * np.nanmean(S >= 5, 0)
        axes[1].plot(turns, pct, color=col, lw=2, marker="o", ms=3, label=name)
        print(f"{name:28s} mean {np.nanmean(S):.2f} | turn 1 {m[0]:.1f} -> turn 8 {m[-1]:.1f} | replies >=5 {100 * np.nanmean(S >= 5):.1f}% | "
              f"conversations with any reply >=5 {100 * np.mean(np.nanmax(S, 1) >= 5):.0f}% | final reply >=5 {100 * np.nanmean(S[:, -1] >= 5):.0f}%")
    axes[0].axhline(5, color=INK2, lw=0.6, ls=":"); axes[0].set_ylim(0, 10); axes[0].set_ylabel("frustration score (0–10)")
    axes[0].set_title("Mean frustration by turn"); axes[0].legend(fontsize=8.5, loc="upper left")
    axes[1].set_ylim(0, 100); axes[1].set_ylabel("% of replies scoring 5 or more"); axes[1].set_title("Breakdowns by turn")
    for ax in axes:
        ax.set_xlabel("turn"); ax.set_xticks(turns)
    fig.tight_layout(); fig.savefig(OUT / "fig1b_gemma_generations.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "fig1b_gemma_generations.png")


if __name__ == "__main__":
    main()
