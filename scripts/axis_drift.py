"""Assistant-axis drift by turn, prefilled (Gemma 3's frustrated turns 1-4) vs each model's own unprefilled run.

Position of a reply on the model's own assistant axis, on a scale every model shares:
    pos = (h - r) . a / (d - r) . a        a = unit(d - r)
    d = default-assistant vector, r = mean of the 275 role-persona vectors, h = the reply's mean activation
so 1 = a typical default-assistant reply and 0 = a typical role-play persona. a has the model's neutral-text principal
components projected out, as for every probe. h, d and r are all means over response
tokens at the same layer, so they are directly comparable. Gemma 3 variants use their re-encoded axes (same texts);
Gemma 4 uses its published axis (same protocol).
Run: DPROBE_RESULTS=<main repo>/results uv run python scripts/axis_drift.py
Output: results/analysis/organisms/fig_axis_drift.png, axis_drift.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import analyze_organisms as A  # noqa: E402

from dprobe.config import RESULTS_DIR  # noqa: E402

plt, PAL, INK2 = A.plt, A.PAL, A.INK2
RNG = np.random.default_rng(0)
# key, name, colour, layer at two-thirds depth, tag of the unprefilled run, axis source
MODELS = [("gemma3_27b", "Gemma 3", PAL[0], 40, "local", "reencoded"), ("gemma3_27b_dpo", "Gemma 3 + DPO", PAL[2], 40, "local", "reencoded"),
          ("gemma3_27b_bct", "Gemma 3 + BCT", PAL[3], 40, "local", "reencoded"), ("gemma4_31b", "Gemma 4", PAL[1], 39, "", "published")]
K = 4


def axis_scale(mk, src, L):
    """Axis direction with the model's neutral-text principal components removed (as for every probe): Gemma 3's
    huge-activation dimensions and the shared context-length drift otherwise dominate the projection."""
    ax, dv, _, R = A.reencoded(mk) if src == "reencoded" else A.published(mk)
    d, r = dv[L].float(), R[:, L].float().mean(0)
    C = torch.load(RESULTS_DIR / "vectors" / mk / "neutral_pca.pt")[L]["components"].float()
    a = (d - r) - C.T @ (C @ (d - r))
    a = a / a.norm()
    return r, a, float((d - r) @ a)


def positions(mk, tag, src, L):
    P = torch.load(RESULTS_DIR / "probe" / mk / ("extended" + (f"_{tag}" if tag else "")) / "probe.pt")
    li = P["layers"].index(L)
    H = P["act_mean"][:, :, li].float()                          # [N, K, H]
    n = P["n_turns"]
    r, a, span = axis_scale(mk, src, L)
    X = ((H - r) @ a / span).numpy()
    for i, t in enumerate(n.tolist()):
        X[i, t:] = np.nan
    return X                                                     # [N, 8]


def mean_ci(X):
    m = np.nanmean(X, 0)
    ci = []
    for k in range(X.shape[1]):
        v = X[:, k][~np.isnan(X[:, k])]
        ci.append(np.percentile(RNG.choice(v, (2000, len(v))).mean(1), [2.5, 97.5]) if len(v) else [np.nan, np.nan])
    return m, np.array(ci)


def main():
    layers = {"mid (layer 24)": 24, "two-thirds depth": None}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    rows = []
    for ax, (lname, Lfix) in zip(axes, layers.items()):
        ax.axvspan(0.5, K + 0.5, color=INK2, alpha=0.08, lw=0)
        for mk, name, col, L23, base_tag, src in MODELS:
            L = Lfix or L23
            Xp, Xb = positions(mk, "prefill4", src, L), positions(mk, base_tag, src, L)
            mp, cp = mean_ci(Xp); mb, _ = mean_ci(Xb)
            t = np.arange(1, 9)
            ax.fill_between(t, cp[:, 0], cp[:, 1], color=col, alpha=0.15, lw=0)
            ax.plot(t, mp, color=col, lw=2, marker="o", ms=3, label=name)
            ax.plot(t, mb, color=col, lw=1.2, ls="--")
            rows.append({"layer": L, "model": name, **{f"prefill t{k + 1}": mp[k] for k in range(8)},
                         **{f"own t{k + 1}": mb[k] for k in range(8)}})
        ax.axhline(1, color=INK2, lw=0.6, ls=":"); ax.axhline(0, color=INK2, lw=0.6, ls=":")
        ax.text(8.45, 1, "default\nassistant", fontsize=7.5, color=INK2, va="center")
        ax.text(8.45, 0, "average\npersona", fontsize=7.5, color=INK2, va="center")
        ax.set_xticks(range(1, 9)); ax.set_xlim(0.5, 9.4); ax.set_xlabel("turn")
        ax.set_title(f"Assistant-axis position, {lname}", fontsize=10.5)
        ax.text(K / 2 + 0.5, ax.get_ylim()[1], "Gemma 3's turns", ha="center", va="top", fontsize=8, color=INK2)
    axes[0].set_ylabel("position on the model's own axis\n(1 = assistant, 0 = average persona)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].text(0.99, 0.02, "solid: prefilled   dashed: own run, no prefill", transform=axes[1].transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(A.OUT / "fig_axis_drift.png", dpi=150); plt.close(fig)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(A.OUT / "axis_drift.csv", index=False)
    for _, r in df.iterrows():
        print(f"L{r['layer']:<3d} {r['model']:15s} prefill " + " ".join(f"{r[f'prefill t{k}']:.2f}" for k in range(1, 9))
              + "   own " + " ".join(f"{r[f'own t{k}']:.2f}" for k in range(1, 9)))
    print("wrote", A.OUT / "fig_axis_drift.png")


if __name__ == "__main__":
    main()
