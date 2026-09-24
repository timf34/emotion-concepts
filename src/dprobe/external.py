"""External directions imported into a model's vector set (currently: the Assistant Axis).

Source: Tim's precomputed Assistant Axis vectors (HF dataset timf34/gemma-assistant-axis-vectors,
Lu et al. 2026 protocol), shaped [n_layers, d_model] with index = decoder block. Verified: axis =
default-assistant vector − mean of the 275 role vectors, so +axis = more assistant-like.

We (1) project out the same neutral PCs used for the emotion vectors, and (2) rescale each layer's
axis to the median norm of that model's denoised emotion vectors, so a steering multiplier means the
same thing for the axis as for an emotion ("one story-difference"). The scale factors are saved.

Output: RESULTS_DIR/vectors/<model>/vectors_external_dn.pt  {"assistant_axis": {layer: [H]}} (+ _raw, meta)
"""

from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download

from dprobe.config import get_model
from dprobe.extract import load_vectors, vectors_dir

AXIS_REPO = "timf34/gemma-assistant-axis-vectors"
AXIS_DIRS = {"gemma3_27b": "gemma-3-27b", "gemma3_27b_pt": "gemma-3-27b", "gemma4_31b": "gemma-4-31b"}


def import_assistant_axis(model_key: str) -> dict:
    spec = get_model(model_key)
    sub = AXIS_DIRS[model_key.replace("_smoke", "")]
    path = hf_hub_download(AXIS_REPO, f"vectors/{sub}/assistant_axis.pt", repo_type="dataset")
    ax = torch.load(path, map_location="cpu").float()                      # [L, H], index = block
    assert ax.shape == (spec.n_blocks, spec.hidden), f"axis shape {tuple(ax.shape)} != {(spec.n_blocks, spec.hidden)}"
    out = vectors_dir(model_key)
    pca = torch.load(out / "neutral_pca.pt")
    emo = load_vectors(model_key, "emotions", denoised=True)
    raw, dn, scale = {}, {}, {}
    for l, comp in pca.items():
        a = ax[l]
        C = comp["components"]
        a_dn = a - C.T @ (C @ a)
        med = torch.stack([emo[e][l] for e in emo]).norm(dim=1).median().item()
        f = med / (a_dn.norm().item() + 1e-9)
        raw[l], dn[l], scale[l] = a, a_dn * f, f
    torch.save({"assistant_axis": raw}, out / "vectors_external_raw.pt")
    torch.save({"assistant_axis": dn}, out / "vectors_external_dn.pt")
    meta = {"source": f"{AXIS_REPO}/vectors/{sub}/assistant_axis.pt", "sign": "+ = more assistant-like",
            "rescaled_to_median_emotion_norm": True, "scale_by_layer": {str(k): v for k, v in scale.items()}}
    with open(out / "external_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[external] {model_key}: assistant_axis imported at {len(dn)} layers; scale factor at L{spec.two_thirds_layer}: {scale[spec.two_thirds_layer]:.2f}")
    return meta
