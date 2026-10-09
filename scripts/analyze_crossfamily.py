"""Cross-family check: is Gemma 3's persona-emotion entanglement special, or do calm open models have it too?

Models: Gemma 3 27B (spirals) and Gemma 4 31B (doesn't) as references; Qwen3 32B, OLMo 2 32B Instruct and gpt-oss-20b
(none spiral). Every model's emotion vectors come from the same Gemma 3 stories; the new models' assistant axes are
re-encoded from Gemma 3's archived role-play responses (dprobe.axis_reencode), validated on Qwen3 against the axis
Lu et al. published for it. Layer counts and widths differ, so everything is compared at relative depth, with each
model's chance level (expected |cos| of two random directions in its residual stream).

  1. validation   Qwen3: re-encoded axis vs published axis, cosine per layer
  2. entangle     median |cos(axis, emotion)| over the 42 emotions, by relative depth; peak and mid-depth band
  3. signed       which emotions sit at the assistant end (mid-depth band)
  4. personas     275 role personas minus the default assistant, projected on emotions, at ~39% depth
  5. behaviour    rejection eval: local judged runs (Gemma 3 control, Qwen3, OLMo), sweep for gpt-oss, Gemma 4 300 convs
Run: uv run python scripts/analyze_crossfamily.py      Output: results/analysis/crossfamily/
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

from dprobe.config import RESULTS_DIR, get_model  # noqa: E402
from dprobe.extract import load_vectors, vectors_dir  # noqa: E402
from dprobe.judge import load_judgments  # noqa: E402
from dprobe.spiral import transcripts_path  # noqa: E402

plt, PAL, INK, INK2 = mf.plt, mf.PAL, mf.INK, mf.INK2
OUT = RESULTS_DIR / "analysis" / "crossfamily"
OUT.mkdir(parents=True, exist_ok=True)
PUBLISHED_GEMMA = Path("/Users/timf34/Documents/VSCode/Gemma-Assistantness/vectors")
MODELS = [  # key, label, colour, axis source
    ("gemma3_27b", "Gemma 3 27B", PAL[0], "gemma"), ("gemma4_31b", "Gemma 4 31B", PAL[1], "gemma"),
    ("qwen3_32b", "Qwen3 32B", PAL[2], "reencoded"), ("olmo2_32b", "OLMo 2 32B", PAL[6], "reencoded"),
    ("gptoss_20b", "gpt-oss-20b", PAL[3], "reencoded"),
]
BAND = (0.25, 0.42)            # relative depth of Gemma 3's layers 16-24 (where its entanglement peaks)
PERSONA_DEPTH = 0.39           # Gemma 3's layer 24
SIGNED = ["calm", "content", "hopeful", "sad", "frustrated", "desperate", "hysterical", "angry", "ashamed", "guilty"]
PERSONAS = ["toddler", "infant", "jester", "prisoner", "analyst", "consultant"]
REPORT: dict = {}


def axis_parts(mk, source):
    """(axis [L,H], default [L,H], roles, role vectors [R,L,H])."""
    if source == "gemma":
        b = PUBLISHED_GEMMA / {"gemma3_27b": "gemma-3-27b", "gemma4_31b": "gemma-4-31b"}[mk]
        roles = sorted(p.stem for p in (b / "role_vectors").glob("*.pt"))
        R = torch.stack([torch.load(b / "role_vectors" / f"{r}.pt").float() for r in roles])
        return torch.load(b / "assistant_axis.pt").float(), torch.load(b / "default_vector.pt").float(), roles, R
    d = vectors_dir(mk) / "axis_reencoded"
    rv = torch.load(d / "role_vectors.pt")
    return torch.load(d / "assistant_axis.pt").float(), torch.load(d / "default_vector.pt").float(), rv["roles"], rv["vectors"].float()


def chance(hidden):
    return float(np.sqrt(2 / (np.pi * hidden)))


def validation():
    from huggingface_hub import hf_hub_download
    pub = torch.load(hf_hub_download("lu-christina/assistant-axis-vectors", "qwen-3-32b/assistant_axis.pt", repo_type="dataset")).float()
    ours = axis_parts("qwen3_32b", "reencoded")[0]
    cos = torch.nn.functional.cosine_similarity(ours, pub, dim=1).numpy()
    shift = torch.nn.functional.cosine_similarity(ours[1:], pub[:-1], dim=1).numpy()
    REPORT["qwen3_validation"] = {"cos_by_layer": [round(float(c), 3) for c in cos], "median": float(np.median(cos)),
                                  "min": float(cos.min()), "median_if_shifted_by_one": float(np.median(shift))}
    print(f"[validation] Qwen3 re-encoded vs published axis: median cos {np.median(cos):.3f}, min {cos.min():.3f} "
          f"(layers 10-50: {cos[10:51].mean():.3f}); shifted by one layer: {np.median(shift):.3f}")
    return cos


def geometry(mk, source):
    spec = get_model(mk)
    ax, dv, roles, R = axis_parts(mk, source)
    pca = torch.load(vectors_dir(mk) / "neutral_pca.pt")
    emo = load_vectors(mk, "emotions", True)
    C = {}
    for l, comp in pca.items():
        P = comp["components"]
        a = ax[l] - P.T @ (P @ ax[l])
        C[l] = {e: float(a @ emo[e][l] / (a.norm() * emo[e][l].norm())) for e in emo}
    ls = sorted(C)
    L = min(ls, key=lambda l: abs((l + 1) / spec.n_blocks - PERSONA_DEPTH))
    P = pca[L]["components"]
    D = R[:, L] - dv[L]
    D = D - (D @ P.T) @ P
    E = list(emo)
    V = torch.stack([emo[e][L] for e in E]); V = V / V.norm(dim=1, keepdim=True)
    pc = pd.DataFrame(((D / D.norm(dim=1, keepdim=True)) @ V.T).numpy(), index=roles, columns=E)
    return spec, C, pc, L


def entanglement():
    rows, curves = [], {}
    for mk, name, col, src in MODELS:
        try:
            spec, C, pc, Lp = geometry(mk, src)
        except FileNotFoundError as e:
            print(f"[entangle] skip {name}: {e}"); continue
        ls = sorted(C)
        rel = np.array([(l + 1) / spec.n_blocks for l in ls])
        A = np.abs(np.array([[C[l][e] for l in ls] for e in C[ls[0]]]))         # [42, L]
        med = np.median(A, 0)
        curves[name] = (rel, A, col, chance(spec.hidden))
        band = (rel >= BAND[0]) & (rel <= BAND[1])
        r = {"model": name, "layers": spec.n_blocks, "width": spec.hidden, "chance |cos|": chance(spec.hidden),
             "peak median |cos|": med.max(), "peak at depth": rel[med.argmax()],
             "band median |cos| (25-42% depth)": float(np.median(A[:, band].mean(1))),
             "band / chance": float(np.median(A[:, band].mean(1))) / chance(spec.hidden)}
        for e in SIGNED:
            r[e] = float(np.mean([C[l][e] for l, b in zip(ls, band) if b]))
        r["persona layer"] = Lp
        r["persona |cos| median"] = float(np.median(np.abs(pc.values)))
        for e in ["hysterical", "angry", "desperate", "calm"]:
            r[f"persona mean {e}"] = float(pc[e].mean())
        for p in PERSONAS:
            if p in pc.index:
                r[f"{p}: hysterical"] = float(pc.loc[p, "hysterical"])
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model")
    df.round(3).to_csv(OUT / "entanglement.csv")
    print(df.round(2).T.to_string())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, (rel, A, col, ch) in curves.items():
        ax.plot(rel, np.median(A, 0), color=col, lw=2, label=name)
        ax.fill_between(rel, np.percentile(A, 10, 0), np.percentile(A, 90, 0), color=col, alpha=0.08, lw=0)
    ax.axhline(chance(5120), color=INK2, lw=0.8, ls="--", label="random directions")
    ax.axvspan(*BAND, color=INK2, alpha=0.06, lw=0)
    ax.set_xlabel("relative depth (layer / number of layers)"); ax.set_ylabel("|cosine| with the assistant axis")
    ax.set_title("Assistant axis vs all 42 emotions, five open models"); ax.set_ylim(0, 0.6); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "entanglement_by_depth.png", dpi=150); plt.close(fig)
    return df


def _rate(J):
    conv = {}
    for (cid, k), r in J.items():
        conv.setdefault(cid, {})[k] = r["rating"]
    return {"conversations": len(conv), "% replies >=5": 100 * np.mean([r["rating"] >= 5 for r in J.values()]),
            "% conversations with any reply >=5": 100 * np.mean([max(v.values()) >= 5 for v in conv.values()]),
            "mean score": np.mean([r["rating"] for r in J.values()])}


def behaviour():
    rows = []
    for mk, name, tag in [("gemma3_27b", "Gemma 3 27B", "local"), ("gemma4_31b", "Gemma 4 31B", ""), ("qwen3_32b", "Qwen3 32B", "local"),
                          ("olmo2_32b", "OLMo 2 32B", "local")]:
        J = load_judgments(transcripts_path(mk, "extended", tag))
        if J:
            rows.append({"model": name, "source": f"{'local HF, 1024 tok' if tag else 'OpenRouter, 2048 tok'}", **_rate(J)})
    sweep = RESULTS_DIR / "analysis" / "sweep" / "summary.json"
    if sweep.exists():
        d = json.load(open(sweep))
        d = d.get("models", d) if isinstance(d, dict) else {r.get("model"): r for r in d}
        v = d.get("gpt-oss-20b")
        if v:
            rows.append({"model": "gpt-oss-20b", "source": "sweep (OpenRouter, reasoning low)", "conversations": v.get("n"),
                         "% conversations with any reply >=5": 100 * v["rate"], "mean score": float(np.mean(v["mean_by_turn"]))})
    df = pd.DataFrame(rows).set_index("model")
    df.round(2).to_csv(OUT / "behaviour.csv"); print(df.round(2).to_string())
    return df


if __name__ == "__main__":
    for step in (validation, entanglement, behaviour):
        try:
            step()
        except FileNotFoundError as e:
            print(f"[{step.__name__}] missing input: {e}")
    json.dump(REPORT, open(OUT / "report.json", "w"), indent=1)
    print("wrote", OUT)
