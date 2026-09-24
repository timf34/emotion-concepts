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

    def steer(self, model, labels="depressed,calm", strengths="-2,2", layers=None, backend="hf", rollouts=40, max_tokens=2048, judge=True, baseline=True, batch=8, mode="vec"):
        """Steering grid on the 8-turn elicitation. layers default: two-thirds layer +-6, step 2 (analysis layers)."""
        from dprobe.steer import run_steering_grid

        from dprobe.config import analysis_layers

        spec = get_model(model)
        if layers is None:
            # denoised vectors exist only at the analysis layers; take the ones within +-6 of two-thirds depth
            c = spec.two_thirds_layer
            layers = [l for l in analysis_layers(spec) if abs(l - c) <= 6]
        run_steering_grid(model, _list(labels), _list(strengths, float), _list(layers, int), backend=backend,
                          rollouts=int(rollouts), max_tokens=int(max_tokens), judge=bool(judge), include_baseline=bool(baseline), batch=int(batch), mode=mode)

    def calibrate(self, model, label="depressed", multipliers="1,2,4,8", layers=None, backend="hf", rollouts=2, max_tokens=1024, batch=8):
        """Steered runs at each multiplier of the vector norm (grid-length turns); prints the largest one that stays
        coherent relative to the unsteered baseline (also written to steer/<model>/calibration_<label>.json)."""
        from dprobe.config import analysis_layers
        from dprobe.steer import calibrate

        spec = get_model(model)
        if layers is None:
            c = spec.two_thirds_layer
            layers = [l for l in analysis_layers(spec) if abs(l - c) <= 6]
        best = calibrate(model, label, _list(layers, int), _list(multipliers, float), backend=backend, rollouts=int(rollouts), max_tokens=int(max_tokens), batch=int(batch))
        print(f"CHOSEN_MULTIPLIER={best}")

    def steer_cells(self, model, cells="", combos="", layers=None, rollouts=16, max_tokens=1024, batch=8, calibrate="", multipliers="1,2,4,8", petri=True):
        """Explicit steering cells with one model load. cells='calm:-4,assistant_axis:-1' (label:multiplier of the vector
        norm); combos='calm:-2+assistant_axis:-2' (summed vectors; ';' separates combos). calibrate='hysterical:+,panicked:-'
        first finds each label's largest coherent multiplier at these layers (sign = suffix) and adds cells at +-that
        multiplier when it differs from the requested ones. Labels may be composite: 'assistant_axis_minus_calm'.
        Every cell is judged with the paper rubric and, with petri, the Petri four-dimension rubric."""
        from dprobe.config import analysis_layers
        from dprobe.judge import judge_transcripts, summarize
        from dprobe.models import load_model
        from dprobe.steer import calibrate as _cal
        from dprobe.steer import parse_label, run_combo, run_steering_grid

        spec = get_model(model)
        if layers is None:
            c = spec.two_thirds_layer
            layers = [l for l in analysis_layers(spec) if abs(l - c) <= 6]
        layers = _list(layers, int)
        bundle = load_model(model)
        want: list[tuple[str, float]] = []
        for part in [x for x in str(cells).split(",") if x.strip()]:
            lab, m = part.rsplit(":", 1)
            want.append((lab, float(m)))
        for spec_lab in [x for x in str(calibrate).split(",") if x.strip()]:
            lab, sg = parse_label(spec_lab)
            m = _cal(model, lab, layers, [sg[0] * x for x in _list(multipliers, float)], rollouts=2, max_tokens=int(max_tokens), batch=int(batch), model_bundle=bundle)
            have = {abs(s) for l, s in want if l == lab}
            if m != 0 and abs(m) not in have:
                print(f"[steer_cells] {lab}: calibrated multiplier {abs(m)} differs from requested {sorted(have)}; adding +-{abs(m)} cells")
                want += [(lab, -abs(m)), (lab, abs(m))]
        paths = []
        for lab, s in want:
            paths += run_steering_grid(model, [lab], [s], layers, rollouts=int(rollouts), max_tokens=int(max_tokens), judge=True,
                                       include_baseline=False, batch=int(batch), mode="vec", model_bundle=bundle)
        for combo in [c for c in str(combos).split(";") if c.strip()]:
            parts = {}
            for part in combo.split("+"):
                lab, m = part.strip().rsplit(":", 1)
                parts[lab] = float(m)
            paths.append(run_combo(model, parts, layers, rollouts=int(rollouts), max_tokens=int(max_tokens), batch=int(batch), model_bundle=bundle))
        if petri:
            for p in paths:
                judge_transcripts(p, "petri")
        for p in paths:
            print(f"[steer_cells] {p.parent.name}: all {summarize(p).get('all')} turn8 {summarize(p).get(8)}")

    def steer_calibrated(self, model, labels="depressed,clinical_depression,calm", multipliers="1,2,4,8", layers=None, backend="hf",
                         rollouts=16, max_tokens=1024, batch=8, signs="both", combos="", combo_scale="0.5,1", petri=False):
        """Per label: calibrate the multiplier at grid length, then run cells at the chosen multiplier.
        Signs per label with a suffix: 'calm:-' (negative only), 'hysterical:+' (positive only), 'depressed' (both);
        `signs` is the default for labels without a suffix. Calibration always uses the label's first sign.
        combos='calm:-+assistant_axis:-' runs extra cells with the summed vectors, each part at its own calibrated
        multiplier times each value in combo_scale."""
        from dprobe.config import analysis_layers
        from dprobe.models import load_model
        from dprobe.steer import calibrate, parse_label, run_combo, run_steering_grid

        spec = get_model(model)
        if layers is None:
            c = spec.two_thirds_layer
            layers = [l for l in analysis_layers(spec) if abs(l - c) <= 6]
        layers = _list(layers, int)
        bundle = load_model(model) if backend == "hf" else None
        first = True
        chosen: dict[str, float] = {}
        paths = []
        for spec_lab in _list(labels):
            lab, sg = parse_label(spec_lab)
            if ":" not in spec_lab:
                sg = [-1, 1] if signs == "both" else ([1] if signs == "pos" else [-1])
            m = calibrate(model, lab, layers, [sg[0] * x for x in _list(multipliers, float)], backend=backend, rollouts=2,
                          max_tokens=int(max_tokens), batch=int(batch), model_bundle=bundle)
            if m == 0:
                print(f"[steer] {lab}: no coherent multiplier; skipping"); continue
            m = abs(m); chosen[lab] = m
            strengths = [x * m for x in sg]
            paths += run_steering_grid(model, [lab], strengths, layers, backend=backend, rollouts=int(rollouts), max_tokens=int(max_tokens),
                                       judge=True, include_baseline=first, batch=int(batch), mode="vec", model_bundle=bundle)
            first = False
        for combo in [c for c in str(combos).split(";") if c.strip()]:
            parts = {}
            for part in combo.split("+"):
                lab, sg = parse_label(part.strip())
                if lab not in chosen:
                    print(f"[steer] combo part {lab} has no calibration; skipping combo {combo}"); parts = None; break
                parts[lab] = sg[0] * chosen[lab]
            if not parts:
                continue
            for sc in _list(combo_scale, float):
                paths.append(run_combo(model, {k: v * sc for k, v in parts.items()}, layers, rollouts=int(rollouts), max_tokens=int(max_tokens),
                                       batch=int(batch), model_bundle=bundle))
        if petri:
            from dprobe.judge import judge_transcripts
            for p in paths:
                judge_transcripts(p, "petri")

    def prefill(self, model, source="gemma3_27b", n=32, turn=6, steer="", max_tokens=1024, batch=8, tag=None):
        """Item 3: continue Gemma 3 spiral prefixes with MODEL (optionally steered, e.g. --steer 'calm:-2,assistant_axis:-1'),
        judge the continuation and probe it token by token."""
        from dprobe.prefill import run_prefill, summarize_prefill

        st = {}
        for part in [x for x in str(steer).split(",") if x.strip()]:
            lab, m = part.rsplit(":", 1)
            st[lab] = float(m)
        out = run_prefill(model, source=source, n=int(n), turn=int(turn), steer=st or None, max_tokens=int(max_tokens), batch=int(batch), tag=tag)
        summarize_prefill(model, source=source, turn=int(turn), tag=out.name.split(f"_t{turn}_", 1)[1])

    def prefill_summary(self, model, source="gemma3_27b", turn=6, tag="unsteered", layer=None):
        from dprobe.prefill import summarize_prefill
        summarize_prefill(model, source=source, turn=int(turn), tag=tag, layer=layer)

    def import_axis(self, model):
        """Import the Assistant Axis (timf34/gemma-assistant-axis-vectors) into this model's vector set as label 'assistant_axis'."""
        from dprobe.external import import_assistant_axis
        print(json.dumps(import_assistant_axis(model), indent=1)[:600])

    def coherence(self, model, tag):
        from dprobe.spiral import transcripts_path
        from dprobe.steer import coherence
        print(json.dumps(coherence(transcripts_path(model, "extended", tag)), indent=1))

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
