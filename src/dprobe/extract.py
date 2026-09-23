"""Emotion / syndrome vector extraction (Anthropic 2026 method), GPU side.

For each story set:
  1. mean residual-stream activation per (label, layer), averaged over tokens >= start_at_nth_token
  2. vector[label] = mean[label] - mean over labels in the same set          (raw, all layers)
  3. project out the top PCs of neutral-story token activations (50% variance)  (denoised, analysis layers)
  4. neutral projection statistics per (label, layer) for z-scoring probe readouts

Files under RESULTS_DIR/vectors/<model_key>/:
  mean_acts_<set>.pt      {label: {layer: [H]}}
  vectors_<set>_raw.pt    {label: {layer: [H]}}
  vectors_<set>_dn.pt     {label: {layer: [H]}}   (analysis layers only)
  neutral_pca.pt          {layer: {"mean": [H], "components": [k, H], "explained": float}}
  neutral_stats.pt        {set: {layer: {"mean": [E], "std": [E], "labels": [...]}}}  (dot-product stats)
  gguf/<label>.gguf       repeng-style control vectors ("direction.<layer>") for EasySteer
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dprobe.config import (
    EMOTIONS, RESULTS_DIR, SMOKE_EMOTIONS, SMOKE_NEUTRAL, SMOKE_STORIES_PER_LABEL, SMOKE_SYNDROMES, SYNDROMES,
    ExtractConfig, analysis_layers, get_model,
)
from dprobe.models import capture, load_model
from dprobe.stories import load_story_set


def vectors_dir(model_key: str) -> Path:
    d = RESULTS_DIR / "vectors" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


@torch.no_grad()
def mean_activations(model, tok, texts: list[str], layers: list[int], cfg: ExtractConfig, collect_tokens: int = 0, per_story: bool = False):
    """Mean over content tokens (>= start_at_nth_token) of every story, per layer.

    Returns (means, n_tokens, samples[, per_story_means]).
    collect_tokens > 0: also return a random subsample of per-token activations {layer: [n, H]} (neutral PCA).
    per_story=True:     also return per-story mean activations {layer: [n_stories, H] bf16} (held-out validation).
    """
    sums: dict[int, torch.Tensor] = {}
    counts = 0
    sample_buf: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    story_buf: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    keep_prob = None
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(texts), cfg.batch_size), desc="extract", leave=False):
        batch = texts[i : i + cfg.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_length).to(device)
        attn = enc["attention_mask"]
        with capture(model, layers) as acts:
            model(**enc)
        for l in layers:
            h = acts[l].float()
            for b in range(h.shape[0]):
                T = int(attn[b].sum())
                if T <= cfg.start_at_nth_token + 1:
                    continue
                x = h[b, cfg.start_at_nth_token:T]
                if l not in sums:
                    sums[l] = torch.zeros(x.shape[-1], device=device, dtype=torch.float64)
                sums[l] += x.sum(0).double()
                if l == layers[0]:
                    counts += x.shape[0]
                if per_story:
                    story_buf[l].append(x.mean(0).to(torch.bfloat16).cpu())
                if collect_tokens:
                    if keep_prob is None:
                        est_total = max(1, len(texts) * max(1, T - cfg.start_at_nth_token))
                        keep_prob = min(1.0, collect_tokens / est_total)
                    m = torch.rand(x.shape[0], device=device) < keep_prob
                    if m.any():
                        sample_buf[l].append(x[m].cpu())
    means = {l: (sums[l] / counts).float().cpu() for l in sums}
    samples = {l: torch.cat(v, 0) if v else None for l, v in sample_buf.items()} if collect_tokens else None
    if per_story:
        return means, counts, samples, {l: torch.stack(v) if v else None for l, v in story_buf.items()}
    return means, counts, samples


def _center(mean_acts: dict[str, dict[int, torch.Tensor]]) -> dict[str, dict[int, torch.Tensor]]:
    labels = list(mean_acts)
    layers = list(next(iter(mean_acts.values())))
    grand = {l: torch.stack([mean_acts[e][l] for e in labels]).mean(0) for l in layers}
    return {e: {l: mean_acts[e][l] - grand[l] for l in layers} for e in labels}


def _fit_neutral_pca(samples: dict[int, torch.Tensor], variance: float, device: str | None = None, max_rows: int = 40000, seed: int = 0) -> dict[int, dict]:
    """Top principal components of neutral tokens per layer, enough to explain `variance` of the total.

    Rows are capped at `max_rows` (random subsample), then the PCA is computed on the GPU from the H x H
    covariance with an eigendecomposition: seconds per layer regardless of how many tokens were collected.
    (A thin SVD of the tall matrix was slow when the token sample was large: cuSOLVER's gesvd has a
    CPU-bound phase.)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l, X in tqdm(samples.items(), desc="neutral PCA", leave=False):
        if X is None or X.shape[0] < 100:
            continue
        if X.shape[0] > max_rows:
            X = X[torch.randperm(X.shape[0], generator=g)[:max_rows]]
        Xd = X.to(device=device, dtype=torch.float32)
        mu = Xd.mean(0, keepdim=True)
        Xc = Xd - mu
        n = Xc.shape[0]
        C = (Xc.T @ Xc) / (n - 1)                                  # [H, H]
        evals, evecs = torch.linalg.eigh(C)                        # ascending
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)   # descending
        cum = torch.cumsum(evals, 0) / evals.sum()
        k = int(torch.searchsorted(cum, torch.tensor(variance, device=cum.device)).item()) + 1
        k = max(1, min(k, evecs.shape[1]))
        out[l] = {
            "mean": mu[0].cpu(), "components": evecs[:, :k].T.cpu().contiguous(), "explained": float(cum[k - 1]), "k": k,
            "resid_norm": float(Xd.norm(dim=1).mean()),   # typical token residual norm; steering strengths are fractions of this
        }
        del Xd, Xc, C, evals, evecs
    return out


def _denoise(vectors: dict[str, dict[int, torch.Tensor]], pca: dict[int, dict]) -> dict[str, dict[int, torch.Tensor]]:
    out = {}
    for e, by_l in vectors.items():
        out[e] = {}
        for l, v in by_l.items():
            if l in pca:
                C = pca[l]["components"]           # [k, H]
                out[e][l] = v - C.T @ (C @ v)
    return out


def _neutral_stats(samples: dict[int, torch.Tensor], vectors: dict[str, dict[int, torch.Tensor]]) -> dict[int, dict]:
    labels = list(vectors)
    out = {}
    for l, X in samples.items():
        if X is None or any(l not in vectors[e] for e in labels):
            continue
        V = torch.stack([vectors[e][l] for e in labels])    # [E, H]
        P = X @ V.T                                          # [n, E]
        out[l] = {"labels": labels, "mean": P.mean(0), "std": P.std(0) + 1e-6}
    return out


def heldout_auc(heldout: dict[str, dict[int, torch.Tensor]], vector_sets: dict[str, dict[str, dict[int, torch.Tensor]]]):
    """One-vs-rest AUC of held-out story means projected onto each label's vector, per layer, raw and denoised.

    Positives: the label's own held-out stories. Negatives: every other label's held-out stories.
    Mirrors the layer-curve AUC in the Pain-axis paper and the top-emotion check in emotion-concepts-in-llms.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    labels = [e for e in heldout if heldout[e] is not None]
    rows = []
    for kind, vecs in vector_sets.items():
        layers = sorted(set.intersection(*[set(vecs[e]) for e in labels]))
        for l in layers:
            X = torch.cat([heldout[e][l].float() for e in labels])                 # [n_all, H]
            y_lab = sum([[e] * heldout[e][l].shape[0] for e in labels], [])
            for e in labels:
                v = vecs[e][l]
                s_ = (X @ v).numpy()
                y = np.array([1 if t == e else 0 for t in y_lab])
                rows.append({"kind": kind, "layer": l, "label": e, "auc": float(roc_auc_score(y, s_)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())})
    return pd.DataFrame(rows)


def export_gguf(vectors: dict[str, dict[int, torch.Tensor]], out_dir: Path, hf_id: str):
    """repeng-style control vector files: tensor 'direction.<layer>' per layer, for EasySteer `source=`."""
    try:
        import gguf
    except ImportError:
        print("[extract] gguf not installed; skipping gguf export")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, by_l in vectors.items():
        w = gguf.GGUFWriter(str(out_dir / f"{label.replace(' ', '_')}.gguf"), "controlvector")
        w.add_string("controlvector.model_hint", hf_id)
        w.add_uint32("controlvector.layer_count", len(by_l))
        for l, v in sorted(by_l.items()):
            w.add_tensor(f"direction.{l}", v.float().numpy())
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def run_extract(model_key: str, sets: tuple[str, ...] = ("emotions", "syndromes"), cfg: ExtractConfig | None = None, all_layers: bool = True, model_bundle=None, smoke: bool = False):
    cfg = cfg or ExtractConfig()
    model, tok, spec = model_bundle or load_model(model_key)
    out = vectors_dir(model_key + ("_smoke" if smoke else ""))
    layers_all = list(range(spec.n_blocks)) if all_layers else analysis_layers(spec, cfg)
    layers_an = analysis_layers(spec, cfg)

    # ---- neutral: PCA basis + z-score stats ----
    pca_path = out / "neutral_pca.pt"
    neutral = load_story_set(model_key, "neutral")["neutral"]
    if smoke:
        neutral = neutral[:SMOKE_NEUTRAL]
    print(f"[extract] {model_key}: {len(neutral)} neutral stories")
    neutral_means, _, samples = mean_activations(model, tok, neutral, layers_an, cfg, collect_tokens=cfg.neutral_tokens_for_pca)
    pca = _fit_neutral_pca(samples, cfg.neutral_pca_variance, max_rows=cfg.neutral_tokens_for_pca)
    torch.save(pca, pca_path)
    print(f"[extract] neutral PCA: k per layer = { {l: d['k'] for l, d in pca.items()} }")

    stats: dict[str, dict] = {}
    for set_name in sets:
        stories = load_story_set(model_key, set_name)
        wanted = (EMOTIONS if set_name == "emotions" else SYNDROMES) if not smoke else (SMOKE_EMOTIONS if set_name == "emotions" else SMOKE_SYNDROMES)
        if smoke:
            stories = {e: v[:SMOKE_STORIES_PER_LABEL] for e, v in stories.items()}
        labels = [e for e in wanted if e in stories]
        missing = [e for e in wanted if e not in stories]
        if missing:
            print(f"[extract] {set_name}: missing stories for {missing}")
        mean_acts: dict[str, dict[int, torch.Tensor]] = {}
        ma_path = out / f"mean_acts_{set_name}.pt"
        if ma_path.exists():
            mean_acts = torch.load(ma_path)
        ho_path = out / f"heldout_{set_name}.pt"
        heldout: dict[str, dict[int, torch.Tensor]] = torch.load(ho_path) if ho_path.exists() else {}
        for e in labels:
            if e in mean_acts and all(l in mean_acts[e] for l in layers_all) and e in heldout:
                continue
            # last `heldout_frac` of each label's stories (topic-ordered) never enter the vector
            texts = stories[e]
            n_ho = max(10, int(len(texts) * cfg.heldout_frac))
            train, held = texts[:-n_ho], texts[-n_ho:]
            means, n_tok, _ = mean_activations(model, tok, train, layers_all, cfg)
            mean_acts[e] = means
            _, _, _, per = mean_activations(model, tok, held, layers_an, cfg, per_story=True)
            heldout[e] = per
            print(f"[extract] {set_name}/{e}: {len(train)} train stories ({n_tok} tokens), {len(held)} held out")
            torch.save(mean_acts, ma_path)
            torch.save(heldout, ho_path)
        raw = _center({e: mean_acts[e] for e in labels})
        dn = _denoise({e: {l: raw[e][l] for l in layers_an} for e in labels}, pca)
        torch.save(raw, out / f"vectors_{set_name}_raw.pt")
        torch.save(dn, out / f"vectors_{set_name}_dn.pt")
        stats[set_name] = {"raw": _neutral_stats(samples, {e: {l: raw[e][l] for l in layers_an} for e in labels}), "dn": _neutral_stats(samples, dn)}
        export_gguf(dn, out / "gguf" / set_name, spec.hf_id)
        auc = heldout_auc(heldout, {"raw": {e: {l: raw[e][l] for l in layers_an} for e in labels}, "dn": dn})
        auc.to_csv(out / f"heldout_auc_{set_name}.csv", index=False)
        best = auc[auc.kind == "dn"].groupby("label").auc.max().round(3).to_dict()
        print(f"[extract] saved {set_name}: {len(labels)} vectors, raw@{len(layers_all)} layers, denoised@{len(layers_an)} layers; held-out AUC (best layer, denoised): {best}")
    # z-score statistics for the externally supplied pain axis too (same direction at every layer)
    pain = load_pain_axis(model_key)
    if pain is not None:
        pv = torch.as_tensor(pain["s2_pain_vector"]).float()
        pst = _neutral_stats(samples, {"pain_axis": {l: pv for l in layers_an}})
        stats["pain_axis"] = {"raw": pst, "dn": pst}
    torch.save(stats, out / "neutral_stats.pt")
    with open(out / "meta.json", "w") as f:
        json.dump({"model": spec.hf_id, "n_blocks": spec.n_blocks, "hidden": spec.hidden, "analysis_layers": layers_an, "cfg": cfg.__dict__}, f, indent=1)
    return out


def load_vectors(model_key: str, set_name: str, denoised: bool = True) -> dict[str, dict[int, torch.Tensor]]:
    p = vectors_dir(model_key) / f"vectors_{set_name}_{'dn' if denoised else 'raw'}.pt"
    return torch.load(p)


def load_pain_axis(model_key: str) -> dict[int, torch.Tensor] | None:
    """Pain-axis paper vectors for gemma-3-27b (S2 first-person, mean-over-tokens extraction)."""
    from dprobe.config import DATA_DIR

    tag = {"gemma3_27b": "it", "gemma3_27b_pt": "pt"}.get(model_key.replace("_smoke", ""))
    if tag is None:
        return None
    p = DATA_DIR / "pain_axis" / f"pain_vectors_gemma3_27b_{tag}" / "pain_vectors.pt"
    if not p.exists():
        return None
    obj = torch.load(p, map_location="cpu", weights_only=False)
    return obj


def stack_vectors(sets: dict[str, dict[str, dict[int, torch.Tensor]]], layer: int) -> tuple[list[str], torch.Tensor]:
    """{set: {label: {layer: v}}} -> (labels, [E, H]) at one layer (label prefixed with set for syndromes)."""
    labels, rows = [], []
    for set_name, vecs in sets.items():
        for e, by_l in vecs.items():
            if layer in by_l:
                labels.append(e)
                rows.append(by_l[layer])
    return labels, torch.stack(rows)
