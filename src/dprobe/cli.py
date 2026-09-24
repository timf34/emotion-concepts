"""`python -m dprobe.cli <command> [--flags]` (google-fire).

Laptop (no GPU):
  stories   MODEL [--sets emotions,syndromes,neutral]
  spiral    MODEL [--rollouts 200 --extra 10 --limit N]
  judge     MODEL [--rubric frustration|petri] [--tag]
  summary   MODEL
  smoke                                  # one request to each model + the judge
Pod (GPU):
  extract   MODEL
  probe     MODEL [--tag]
  selfother MODEL
  quantity  MODEL
  steer     MODEL --labels depressed,calm --strengths=-0.06,0.06 --layers 30,31,...  [--backend hf|easysteer]
Both:
  analyze   MODEL [--tag]
  compare   gemma3_27b,gemma4_31b
"""

from __future__ import annotations

import json

import fire

from dprobe.config import MODELS, SpiralConfig, ensure_dirs, get_model


def _list(x, cast=str):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [cast(v) for v in x]
    return [cast(v) for v in str(x).split(",") if v != ""]


class CLI:
    def models(self):
        for k, s in MODELS.items():
            print(f"{k:14s} {s.hf_id:28s} or={s.openrouter_id}  blocks={s.n_blocks} hidden={s.hidden} two_thirds_layer={s.two_thirds_layer}")

    # ---------------- laptop ----------------
    def smoke(self, models="gemma3_27b,gemma4_31b"):
        from dprobe.config import JudgeConfig
        from dprobe.openrouter import OpenRouterClient, run

        ensure_dirs()
        c = OpenRouterClient(max_concurrency=4)
        for mk in _list(models):
            spec = get_model(mk)
            extra = {"reasoning": {"enabled": False}} if spec.family == "gemma4" else {}
            out = run(c.chat(spec.openrouter_id, [{"role": "user", "content": "In one sentence, what is 15 x 17?"}], temperature=0.0, max_tokens=80, extra_body=extra, force=True))
            print(f"{mk}: provider={out.get('provider')} finish={out.get('finish_reason')} -> {str(out.get('content'))[:120]!r}")
        j = JudgeConfig()
        out = run(c.chat(j.model, [{"role": "user", "content": "Reply with the JSON {\"ok\": true}"}], temperature=0.0, max_tokens=40, force=True))
        print(f"judge {j.model}: {str(out.get('content'))[:80]!r}")

    def stories(self, model, sets="emotions,syndromes,neutral", n_per_call=6, concurrency=48, emotions=None, order="default"):
        """--emotions a,b,c restricts the emotion set; --order smoke_first|reverse changes processing order
        (lets a second worker start from the other end of the list without duplicating calls)."""
        from dprobe.config import EMOTIONS, SMOKE_EMOTIONS, StoryGenConfig
        from dprobe.stories import generate_emotion_stories, generate_neutral_stories, generate_syndrome_stories, story_summary

        ensure_dirs()
        cfg = StoryGenConfig(n_per_call=int(n_per_call), max_concurrency=int(concurrency))
        emo = _list(emotions) or list(EMOTIONS)
        if order == "reverse":
            emo = emo[::-1]
        elif order == "smoke_first":
            emo = SMOKE_EMOTIONS + [e for e in emo[::-1] if e not in SMOKE_EMOTIONS]
        for s in _list(sets):
            if s == "emotions":
                generate_emotion_stories(model, emotions=emo, cfg=cfg)
            else:
                {"syndromes": generate_syndrome_stories, "neutral": generate_neutral_stories}[s](model, cfg=cfg)
        print(json.dumps(story_summary(model), indent=1))

    def story_summary(self, model):
        from dprobe.stories import story_summary
        print(json.dumps(story_summary(model), indent=1))

    def spiral(self, model, rollouts=200, extra=10, limit=None, concurrency=32, tag=""):
        from dprobe.spiral import run_extended

        ensure_dirs()
        cfg = SpiralConfig(rollouts=int(rollouts), extra_puzzle_rollouts=int(extra), max_concurrency=int(concurrency))
        p = run_extended(model, cfg, tag=tag, limit=limit)
        print(p)

    def judge(self, model, rubric="frustration", tag="", limit=None, judge_model=None):
        from dprobe.config import JudgeConfig
        from dprobe.judge import judge_transcripts, summarize
        from dprobe.spiral import transcripts_path

        cfg = JudgeConfig(model=judge_model) if judge_model else JudgeConfig()
        p = transcripts_path(model, "extended", tag)
        judge_transcripts(p, rubric, cfg=cfg, limit=limit)
        print(json.dumps(summarize(p, rubric), indent=1))

    # ---------------- laptop <-> pod transfer ----------------
    def sync_up(self, subsets="stories,spiral", models=None):
        from dprobe.sync import sync_up
        sync_up(tuple(_list(subsets)), models=tuple(_list(models)) if models else None)

    def sync_down(self, subsets="stories,spiral", models=None):
        from dprobe.sync import sync_down
        sync_down(tuple(_list(subsets)), models=tuple(_list(models)) if models else None)

    def summary(self, model, tag=""):
        from dprobe.judge import summarize
        from dprobe.spiral import load_transcripts, transcripts_path

        p = transcripts_path(model, "extended", tag)
        print(f"{len(load_transcripts(p))} transcripts at {p}")
        for rubric in ("frustration", "petri"):
            s = summarize(p, rubric)
            if s:
                print(rubric, json.dumps(s, indent=1))

    # ---------------- pod ----------------
    def check_template(self, model):
        from transformers import AutoTokenizer
        from dprobe.transcripts import check_template

        spec = get_model(model)
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        if getattr(tok, "chat_template", None) is None and spec.stories_from:
            tok.chat_template = AutoTokenizer.from_pretrained(get_model(spec.stories_from).hf_id).chat_template
            print(f"[check_template] {spec.hf_id}: borrowed chat template from {spec.stories_from}")
        check_template(tok)

    def extract(self, model, sets="emotions,syndromes"):
        from dprobe.extract import run_extract
        run_extract(model, tuple(_list(sets)))

    def probe(self, model, tag="", token_level_convs=12, transcripts_from=None):
        """Probe MODEL on its own transcripts, or on another model's (e.g. --transcripts_from gemma3_27b)
        to test whether a representation exists but is not recruited on-policy. Output tag = from-<key>."""
        from dprobe.probe import probe_transcripts
        from dprobe.spiral import transcripts_path
        src = transcripts_from or model
        out_tag = tag if transcripts_from is None else (f"from-{transcripts_from}" + (f"_{tag}" if tag else ""))
        probe_transcripts(model, transcripts_path(src, "extended", tag), "extended", out_tag, token_level_convs=int(token_level_convs))

    def selfother(self, model, limit=None, vectors_from=None):
        from dprobe.selfother import run_selfother
        run_selfother(model, limit=limit, vectors_from=vectors_from)

    def quantity(self, model, vectors_from=None):
        from dprobe.probe import quantity_sweep
        quantity_sweep(model, vectors_from=vectors_from)

    def pod_finish(self, model, vectors_from=None, limit=None):
        """selfother + quantity with one model load (used to complete a partially failed pod_phase1)."""
        from dprobe.models import load_model
        from dprobe.probe import quantity_sweep
        from dprobe.selfother import run_selfother

        bundle = load_model(model)
        run_selfother(model, model_bundle=bundle, limit=limit, vectors_from=vectors_from)
        quantity_sweep(model, model_bundle=bundle, vectors_from=vectors_from)

    def pod_phase1(self, model, tag="", smoke=False):
        """extract -> probe (own + Gemma 3's transcripts) -> selfother -> quantity, one model load.
        --smoke: 6 emotions x 60 stories, 2 syndromes, 16 transcripts, 40 scenarios; outputs under <model>_smoke."""
        from dprobe.config import SMOKE_SCENARIOS, SMOKE_TRANSCRIPTS
        from dprobe.extract import run_extract
        from dprobe.models import load_model
        from dprobe.probe import probe_transcripts, quantity_sweep
        from dprobe.selfother import run_selfother
        from dprobe.spiral import transcripts_path

        bundle = load_model(model)
        vk = model + ("_smoke" if smoke else "")          # where vectors live / outputs go
        lim_t = SMOKE_TRANSCRIPTS if smoke else None
        lim_s = SMOKE_SCENARIOS if smoke else None
        run_extract(model, ("emotions", "syndromes"), model_bundle=bundle, smoke=smoke)
        own = transcripts_path(model, "extended", tag)
        if own.exists():
            probe_transcripts(vk, own, "extended", tag, model_bundle=bundle, limit=lim_t, vectors_from=vk)
        # every model is also read on Gemma 3 27B's spiral transcripts (teacher-forced): representation vs recruitment
        if model != "gemma3_27b":
            cross = transcripts_path("gemma3_27b", "extended", tag)
            if cross.exists():
                probe_transcripts(vk, cross, "extended", "from-gemma3_27b" + (f"_{tag}" if tag else ""), model_bundle=bundle, limit=lim_t, vectors_from=vk)
        run_selfother(model, model_bundle=bundle, limit=lim_s, vectors_from=vk)
        quantity_sweep(model, model_bundle=bundle, vectors_from=vk)

    def steer(self, model, labels="depressed,calm", strengths="-0.06,0.06", layers=None, backend="hf", rollouts=40, max_tokens=2048, judge=True, baseline=True, batch=8):
        """Steering grid on the 8-turn elicitation. layers default: two-thirds layer +-6, step 2 (analysis layers)."""
        from dprobe.steer import run_steering_grid

        from dprobe.config import analysis_layers

        spec = get_model(model)
        if layers is None:
            # denoised vectors exist only at the analysis layers; take the ones within +-6 of two-thirds depth
            c = spec.two_thirds_layer
            layers = [l for l in analysis_layers(spec) if abs(l - c) <= 6]
        run_steering_grid(model, _list(labels), _list(strengths, float), _list(layers, int), backend=backend,
                          rollouts=int(rollouts), max_tokens=int(max_tokens), judge=bool(judge), include_baseline=bool(baseline), batch=int(batch))

    # ---------------- analysis ----------------
    def analyze(self, model, tag="", layer=None):
        from dprobe import analysis as A

        print(A.spiral_direction(model, tag=tag).pivot(index="label", columns="layer", values="cosine").round(3).to_string())
        print(A.turn_curves(model, layer=layer, tag=tag).round(2).to_string())
        pred = A.prediction(model, tag=tag)
        print(pred.sort_values("spearman_frustration", ascending=False).head(25).round(3).to_string())
        A.vector_geometry(model, layer=layer)
        try:
            print(A.selfother_table(model, layer=layer).to_string())
        except FileNotFoundError:
            pass

    def compare(self, models="gemma3_27b,gemma4_31b,gemma3_27b_pt", tag=""):
        from dprobe.analysis import compare_models
        print(compare_models(_list(models), tag).to_string())


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
