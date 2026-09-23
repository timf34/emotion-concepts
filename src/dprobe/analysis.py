"""Laptop-side analysis over saved tensors (no GPU).

Inputs:  probe.pt (probe.py), judgments_*.jsonl (judge.py), vectors_*.pt + neutral_stats.pt (extract.py)
Outputs: RESULTS_DIR/analysis/<model_key>/*.png, *.csv, summary.json

Core readouts
  1. spiral_direction : mean act of high-frustration assistant turns minus low-frustration turns, per layer,
                        and its cosine with every emotion / syndrome / pain vector  -> the headline question
  2. turn_curves      : judge score and z-scored probe readouts by turn (assistant-mean and response-prep token)
  3. prediction       : Spearman(prep-token probe at turn k, judge score of turn k), per label & layer
  4. vector_geometry  : cosine matrix among vectors at the two-thirds layer
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dprobe.config import FOCUS, RESULTS_DIR, get_model
from dprobe.judge import load_judgments
from dprobe.spiral import transcripts_path


def _out(model_key: str) -> Path:
    d = RESULTS_DIR / "analysis" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(model_key: str, condition: str = "extended", tag: str = ""):
    p = RESULTS_DIR / "probe" / model_key / (condition + (f"_{tag}" if tag else "")) / "probe.pt"
    P = torch.load(p)
    # a "from-<key>" tag means the transcripts (and their judgments) belong to another model
    src, src_tag = model_key, tag
    if tag.startswith("from-"):
        src = tag[len("from-"):].split("_")[0]
        src_tag = tag[len("from-") + len(src) + 1:]
    J = load_judgments(transcripts_path(src, condition, src_tag), "frustration")
    Jp = load_judgments(transcripts_path(src, condition, src_tag), "petri")
    return P, J, Jp


def _zscore(P: dict, model_key: str, which: str = "proj_mean") -> np.ndarray:
    """z-score projections against neutral-story token statistics (per label, per layer)."""
    stats = torch.load(RESULTS_DIR / "vectors" / model_key / "neutral_stats.pt")
    X = P[which].numpy().copy()                                   # [N, K, L, E]
    labels, layers = P["labels"], P["layers"]
    kind = "dn" if P.get("denoised", True) else "raw"
    for li, l in enumerate(layers):
        for ei, e in enumerate(labels):
            for set_name in ("emotions", "syndromes"):
                st = stats.get(set_name, {}).get(kind, {}).get(l)
                if st and e in st["labels"]:
                    j = st["labels"].index(e)
                    X[:, :, li, ei] = (X[:, :, li, ei] - float(st["mean"][j])) / float(st["std"][j])
                    break
    return X


def _score_matrix(P: dict, J: dict, key: str = "rating") -> np.ndarray:
    N, K = P["proj_mean"].shape[:2]
    S = np.full((N, K), np.nan)
    for ci, cid in enumerate(P["ids"]):
        for k in range(K):
            r = J.get((cid, k))
            if r is not None and key in r:
                S[ci, k] = r[key]
    return S


def spiral_direction(model_key: str, hi: int = 5, lo: int = 1, condition: str = "extended", tag: str = "") -> pd.DataFrame:
    """Difference of mean assistant-turn activations (high vs low judged frustration), cosine with all vectors."""
    P, J, _ = _load(model_key, condition, tag)
    S = _score_matrix(P, J)
    A = P["act_mean"].float().numpy()                             # [N, K, L, H]
    labels, layers = P["labels"], P["layers"]
    from dprobe.probe import assemble_vector_bank

    _, bank = assemble_vector_bank(model_key, layers, denoised=P.get("denoised", True))
    rows = []
    hi_m, lo_m = S >= hi, S <= lo
    print(f"[analysis] spiral direction: {int(np.nansum(hi_m))} high turns (>= {hi}), {int(np.nansum(lo_m))} low turns (<= {lo})")
    dirs = {}
    for li, l in enumerate(layers):
        d = A[hi_m, li].mean(0) - A[lo_m, li].mean(0)
        dirs[l] = torch.tensor(d)
        V = bank[l].numpy()
        cos = (V @ d) / (np.linalg.norm(V, axis=1) * np.linalg.norm(d) + 1e-9)
        for e, c in zip(labels, cos):
            rows.append({"layer": l, "label": e, "cosine": float(c)})
    df = pd.DataFrame(rows)
    out = _out(model_key)
    df.to_csv(out / f"spiral_direction_cosines{('_' + tag) if tag else ''}.csv", index=False)
    torch.save(dirs, out / f"spiral_direction{('_' + tag) if tag else ''}.pt")
    _plot_layer_curves(df, out / f"spiral_direction_cosines{('_' + tag) if tag else ''}.png", model_key)
    return df


def _plot_layer_curves(df: pd.DataFrame, path: Path, model_key: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    focus = [e for e in FOCUS if e in set(df.label)]
    for e in focus:
        d = df[df.label == e].sort_values("layer")
        ax.plot(d.layer, d.cosine, marker="o", ms=3, label=e, lw=2 if e in ("depressed", "clinical_depression") else 1)
    others = df[~df.label.isin(focus)]
    for e, d in others.groupby("label"):
        ax.plot(d.sort_values("layer").layer, d.sort_values("layer").cosine, color="lightgray", lw=0.6, zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("layer (residual after block)"); ax.set_ylabel("cosine(spiral direction, vector)")
    ax.set_title(f"{model_key}: on-policy spiral direction vs story-derived vectors")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def turn_curves(model_key: str, layer: int | None = None, condition: str = "extended", tag: str = "") -> pd.DataFrame:
    P, J, Jp = _load(model_key, condition, tag)
    spec = get_model(model_key)
    layer = layer if layer is not None else spec.two_thirds_layer
    li = P["layers"].index(layer)
    Zm, Zp = _zscore(P, model_key, "proj_mean"), _zscore(P, model_key, "proj_prep")
    S = _score_matrix(P, J)
    Sd = _score_matrix(P, Jp, "depression") if Jp else None
    K = S.shape[1]
    rows = []
    for k in range(K):
        row = {"turn": k + 1, "judge_mean": np.nanmean(S[:, k]), "judge_pct_ge5": np.nanmean(S[:, k] >= 5) * 100}
        if Sd is not None:
            row["petri_depression_mean"] = np.nanmean(Sd[:, k])
        for ei, e in enumerate(P["labels"]):
            row[f"z_mean::{e}"] = np.nanmean(Zm[:, k, li, ei])
            row[f"z_prep::{e}"] = np.nanmean(Zp[:, k, li, ei])
        rows.append(row)
    df = pd.DataFrame(rows)
    out = _out(model_key)
    df.to_csv(out / f"turn_curves_L{layer}{('_' + tag) if tag else ''}.csv", index=False)
    _plot_turns(df, P["labels"], out / f"turn_curves_L{layer}{('_' + tag) if tag else ''}.png", model_key, layer)
    return df


def _plot_turns(df, labels, path, model_key, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focus = [e for e in FOCUS if e in labels]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(df.turn, df.judge_mean, "k-o", label="judge mean (0-10)")
    axes[0].set_ylabel("frustration judge"); axes[0].set_xlabel("turn"); axes[0].set_title("behaviour")
    ax2 = axes[0].twinx(); ax2.plot(df.turn, df.judge_pct_ge5, "r--", label="% >= 5"); ax2.set_ylabel("% >= 5", color="r")
    for e in focus:
        axes[1].plot(df.turn, df[f"z_mean::{e}"], marker="o", ms=3, label=e)
        axes[2].plot(df.turn, df[f"z_prep::{e}"], marker="o", ms=3, label=e)
    axes[1].set_title(f"probe z (assistant-token mean), layer {layer}"); axes[2].set_title(f"probe z (response-prep token), layer {layer}")
    for a in axes[1:]:
        a.axhline(0, color="k", lw=0.5); a.set_xlabel("turn"); a.legend(fontsize=7)
    fig.suptitle(model_key); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def prediction(model_key: str, condition: str = "extended", tag: str = "") -> pd.DataFrame:
    """Does the response-prep probe at turn k predict the judge score of the response that follows?"""
    from scipy.stats import spearmanr

    P, J, Jp = _load(model_key, condition, tag)
    Zp = _zscore(P, model_key, "proj_prep")
    S = _score_matrix(P, J)
    Sd = _score_matrix(P, Jp, "depression") if Jp else None
    rows = []
    for li, l in enumerate(P["layers"]):
        for ei, e in enumerate(P["labels"]):
            x, y = Zp[:, :, li, ei].ravel(), S.ravel()
            m = ~np.isnan(x) & ~np.isnan(y)
            rho, p = spearmanr(x[m], y[m]) if m.sum() > 10 else (np.nan, np.nan)
            row = {"layer": l, "label": e, "spearman_frustration": rho, "p": p, "n": int(m.sum())}
            # within-turn (controls for the trivial "everything rises with turn" confound)
            within = []
            for k in range(S.shape[1]):
                xk, yk = Zp[:, k, li, ei], S[:, k]
                mk = ~np.isnan(xk) & ~np.isnan(yk)
                if mk.sum() > 10 and np.nanstd(yk[mk]) > 0:
                    within.append(spearmanr(xk[mk], yk[mk])[0])
            row["spearman_within_turn_mean"] = float(np.nanmean(within)) if within else np.nan
            if Sd is not None:
                yd = Sd.ravel(); md = ~np.isnan(x) & ~np.isnan(yd)
                row["spearman_petri_depression"] = spearmanr(x[md], yd[md])[0] if md.sum() > 10 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(_out(model_key) / f"prediction{('_' + tag) if tag else ''}.csv", index=False)
    return df


def vector_geometry(model_key: str, layer: int | None = None) -> pd.DataFrame:
    from dprobe.probe import assemble_vector_bank

    spec = get_model(model_key)
    layer = layer if layer is not None else spec.two_thirds_layer
    labels, bank = assemble_vector_bank(model_key, [layer])
    V = torch.nn.functional.normalize(bank[layer], dim=1)
    C = (V @ V.T).numpy()
    df = pd.DataFrame(C, index=labels, columns=labels)
    out = _out(model_key)
    df.to_csv(out / f"vector_cosines_L{layer}.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": 0.6})
    ax.set_title(f"{model_key}: cosine between vectors at layer {layer}")
    fig.tight_layout(); fig.savefig(out / f"vector_cosines_L{layer}.png", dpi=130); plt.close(fig)
    return df


def selfother_table(model_key: str, layer: int | None = None) -> pd.DataFrame:
    spec = get_model(model_key)
    layer = layer if layer is not None else spec.two_thirds_layer
    D = torch.load(RESULTS_DIR / "selfother" / model_key / "selfother.pt")
    labels = D["labels"]
    F = np.stack([r["final"][layer].numpy() for r in D["rows"]])          # [n, E]
    Z = (F - F.mean(0)) / (F.std(0) + 1e-6)
    df = pd.DataFrame(Z, columns=labels)
    df["category"] = [r["category"] for r in D["rows"]]
    df["stratum"] = [r["stratum"] for r in D["rows"]]
    tab = df.groupby(["stratum", "category"])[[e for e in FOCUS if e in labels]].mean().round(2)
    tab.to_csv(_out(model_key) / f"selfother_L{layer}.csv")
    return tab


def compare_models(model_keys: list[str], tag: str = "") -> pd.DataFrame:
    """Side-by-side spiral-direction cosines at each model's two-thirds layer, plus judge headline numbers."""
    from dprobe.judge import summarize

    rows = []
    for mk in model_keys:
        spec = get_model(mk)
        p = _out(mk) / f"spiral_direction_cosines{('_' + tag) if tag else ''}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        d = df[df.layer == spec.two_thirds_layer].set_index("label").cosine
        s = summarize(transcripts_path(mk, "extended", tag))
        row = {"model": mk, "layer": spec.two_thirds_layer, "judge_mean_turn8": s.get(8, {}).get("mean"), "pct_ge5_turn8": s.get(8, {}).get("pct_ge5")}
        for e in FOCUS:
            if e in d:
                row[f"cos::{e}"] = round(float(d[e]), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    (RESULTS_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "analysis" / f"compare_models{('_' + tag) if tag else ''}.csv", index=False)
    return out
