"""HF model loading + residual-stream capture that works for Gemma 3 / Gemma 4 wrapper classes.

Gemma 3 and Gemma 4 checkpoints load as *ForConditionalGeneration (multimodal) classes whose
decoder blocks live under a language_model submodule, so the usual `model.model.layers`
lookup fails. `decoder_layers()` walks the known paths. Activations are captured with forward
hooks on the selected blocks only (never `output_hidden_states=True`), so 12k-token spiral
transcripts do not materialise 62 layers of hidden states.

dtype is always bfloat16: Gemma 2/3 overflow in float16 and have huge-activation dimensions.
"""

from __future__ import annotations

import contextlib
from typing import Iterable

import torch
from torch import nn

from dprobe.config import ModelSpec, get_model

_LAYER_PATHS = (
    "model.language_model.layers",      # Gemma3/4ForConditionalGeneration (transformers >= 4.52 / 5.x)
    "language_model.model.layers",      # older multimodal layouts
    "model.layers",                     # plain causal LMs (Gemma3ForCausalLM, Qwen, Llama)
    "model.text_model.layers",
    "transformer.h",
)


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in _LAYER_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"cannot find decoder layers on {type(model).__name__}; tried {_LAYER_PATHS}")


def text_config(model: nn.Module):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def load_model(model_key: str, device: str = "cuda", attn: str = "sdpa"):
    """Returns (model, tokenizer, spec). bf16 on GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model(model_key)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if getattr(tok, "chat_template", None) is None and spec.stories_from:
        # base checkpoints ship no chat template; borrow the instruct model's (identical vocabulary and
        # turn tokens) so transcripts render exactly as the instruct model saw them
        donor = get_model(spec.stories_from)
        tok.chat_template = AutoTokenizer.from_pretrained(donor.hf_id).chat_template
        print(f"[models] {spec.hf_id} has no chat template; using {donor.hf_id}'s")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, device_map=device, attn_implementation=attn
    )
    model.eval()
    layers = decoder_layers(model)
    n = len(layers)
    hidden = text_config(model).hidden_size
    if n != spec.n_blocks or hidden != spec.hidden:
        print(f"[models] WARNING registry says {spec.n_blocks}x{spec.hidden} but model has {n}x{hidden}; trusting the model")
        spec = ModelSpec(spec.key, spec.hf_id, spec.openrouter_id, n, hidden, spec.family, spec.is_base, spec.stories_from)
    print(f"[models] loaded {spec.hf_id}: {n} blocks, hidden={hidden}, class={type(model).__name__}, attn={attn}")
    return model, tok, spec


@contextlib.contextmanager
def capture(model: nn.Module, layers: Iterable[int]):
    """Context manager: `with capture(model, [20, 40]) as acts: model(**enc)` -> acts[layer] = [B, T, H] (bf16, on device)."""
    acts: dict[int, torch.Tensor] = {}
    blocks = decoder_layers(model)
    handles = []

    def mk(l):
        def hook(_m, _i, out):
            acts[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for l in layers:
        handles.append(blocks[l].register_forward_hook(mk(l)))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def add_vectors(model: nn.Module, vec_by_layer: dict[int, torch.Tensor], positions: str = "all", prompt_len: int | None = None):
    """Steering hook: add vec_by_layer[l] (already scaled, [H]) to block l's output.

    positions: "all" (prompt + generation) | "generation" (only tokens after prompt_len, and every
    decode step) | "prompt" (only the prefill pass).
    """
    blocks = decoder_layers(model)
    handles = []

    def mk(v):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            T = h.shape[1]
            vv = v.to(device=h.device, dtype=h.dtype)
            if positions == "all":
                h = h + vv
            elif positions == "generation":
                if T == 1:                       # decode step
                    h = h + vv
                elif prompt_len is not None and T > prompt_len:
                    h = h.clone()
                    h[:, prompt_len:] += vv
            elif positions == "prompt":
                if T > 1:
                    h = h + vv
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    for l, v in vec_by_layer.items():
        handles.append(blocks[l].register_forward_hook(mk(v)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
