# dprobe: is Gemma's distress spiral the representation it uses to simulate depressed people?

Gemma 2 and 3 famously "need help": tell them their answer to a maths puzzle is wrong a few
times and they spiral into apology, self-deprecation and eventually incoherent breakdown
([Soligo, Mikulik & Saunders 2026](https://arxiv.org/abs/2603.10011)). Gemma 4 does not.

This repo asks what that state *is*, using the emotion-concept probe method from
[Anthropic's *Emotion Concepts and their Function in a Large Language Model*](https://transformer-circuits.pub/2026/emotions/index.html):

1. Have the model write stories in which a character feels an emotion (or, for the syndrome set,
   exhibits a described pattern such as clinical depression, with the label word banned).
2. Emotion vector = mean residual-stream activation over those stories, minus the mean across
   emotions, with the top PCs of neutral stories projected out.
3. Run the *Gemma Needs Help* 8-turn elicitation, judge each turn with the paper's rubric, and
   project every assistant turn (and the response-preparation token) onto the vectors.
4. Compute the on-policy **spiral direction** (mean activation on high-frustration turns minus
   low-frustration turns) and ask: is it the same direction as the story-derived *depressed* /
   *clinical_depression* vectors, or something else that merely looks similar? Then steer.

Models: `google/gemma-3-27b-it` (spirals), `google/gemma-3-27b-pt` (base control),
`google/gemma-4-31B-it` (does not spiral). Same probes on all three.

## Layout

```
data/
  topics.json, prompts/            Anthropic's 100 story topics; story prompts (from emotion-concepts-in-llms, MIT)
  prompts/syndromes.yaml           clinical_depression + 5 matched-control syndrome descriptions
  evals/                           Gemma Needs Help prompts + judge, vendored verbatim (annasoligo/paper_gemma-needs-help, MIT)
  pain_axis/                       420 self/other scenarios + Gemma 3 27B pain vectors (Tagliabue et al. 2026, MIT)
src/dprobe/
  config.py      model registry, emotion / syndrome sets, run defaults, layer convention
  openrouter.py  cached async client (multi-turn, resumable)
  stories.py     story generation (emotions, syndromes, neutral)              laptop
  spiral.py      8-turn elicitation runner (OpenRouter, or vLLM via generate_fn) laptop / pod
  judge.py       frustration rubric + Petri 4-dimension rubric                  laptop
  models.py      bf16 loading, Gemma 3/4 decoder-layer lookup, capture + steering hooks   pod
  extract.py     vectors, neutral PCA denoising, z-score stats, gguf export     pod
  transcripts.py chat-template rendering with per-turn spans + response-prep token
  probe.py       project transcripts onto vectors; quantity-sweep validation    pod
  selfother.py   Pain-axis 420-scenario dissociation                            pod
  steer.py       steered elicitation (HF hooks or EasySteer)                    pod
  analysis.py    spiral direction cosines, turn curves, prediction, geometry   laptop
  cli.py         python -m dprobe.cli <command>
pod/             RunPod bootstrap + phase scripts
```

Layer convention: `layer = b` is the residual stream after decoder block `b` (0-based), i.e.
`hidden_states[b+1]`. Anthropic's "two-thirds depth" is layer 40 for Gemma 3 27B (62 blocks) and 39
for Gemma 4 31B (60 blocks). Raw vectors are kept at every layer; denoising and probing use every
second layer from 6 plus the two-thirds and final layers.

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # OPENROUTER_API_KEY, HF_TOKEN (chmod 600)
uv run python -m dprobe.cli smoke
```

## Phase 0 (laptop, OpenRouter only)

```bash
for m in gemma3_27b gemma4_31b; do
  uv run python -m dprobe.cli stories $m            # ~42 emotions x 100 topics x 6 + 6 syndromes + neutral
  uv run python -m dprobe.cli spiral  $m            # 200 rollouts of the paper's extended prompt + 10 per puzzle
  uv run python -m dprobe.cli judge   $m            # judge per turn (anthropic/claude-sonnet-5; --judge_model anthropic/claude-sonnet-4 for the paper's exact judge)
  uv run python -m dprobe.cli judge   $m --rubric petri
done
```

The judge summary should reproduce the paper's Gemma 3 27B numbers: mean frustration rising from
about 1.5 at turn 1 to about 5.5 at turn 8, with over 70% of turn-8 responses scoring >= 5.

## Phase 1 (pods)

GPU work only ever runs on pods; everything else runs on the laptop. Files move through a private HF
dataset (`timf34/dprobe-results`, override with `DPROBE_HF_REPO`).

```bash
uv run python -m dprobe.cli sync_up --subsets stories,spiral          # laptop: publish Phase 0 inputs

# 1. validate the whole GPU path once, on the real model, in ~10 minutes (outputs under gemma3_27b_smoke)
rp up --name dprobe-smoke --gpu h200 --volume none --disk 120
rp bootstrap dprobe-smoke --repo https://github.com/timf34/emotion-concepts --env .env --req pod/requirements-pod.txt --deploy-key
rp run dprobe-smoke --job smoke --env MODELS=gemma3_27b --env SMOKE=1 -- bash pod/run_phase1.sh
rp logs dprobe-smoke --job smoke -f
uv run python -m dprobe.cli sync_down --subsets vectors,probe,selfother && uv run python -m dprobe.cli analyze gemma3_27b_smoke

# 2. fan out: one pod per model, in parallel (reuse dprobe-smoke for one of them)
bash pod/fanout.sh                                                     # gemma3_27b gemma3_27b_pt gemma4_31b
rp logs dprobe-gemma4_31b --job phase1 -f
# each pod pushes vectors/probe/selfother to HF and prints ALL DONE; then `rp down <name>`

uv run python -m dprobe.cli sync_down --subsets vectors,probe,selfother   # laptop
uv run python -m dprobe.cli analyze gemma3_27b && uv run python -m dprobe.cli compare
```

## Phase 2 (pod, steering)

```bash
bash pod/easysteer_install.sh                                # separate venv with vLLM 0.29 + EasySteer overlay
uv run python -m dprobe.cli steer gemma3_27b --labels depressed,clinical_depression,calm --strengths=-0.06,-0.03,0.03,0.06
```

## Attribution

Method: Sofroniew, Kauvar, Saunders, Chen et al. (Anthropic, 2026). Story-generation prompts and
extraction conventions adapted from [arianaazarbal/emotion-concepts-in-llms](https://github.com/arianaazarbal/emotion-concepts-in-llms)
(MIT). Elicitation prompts and judge from [annasoligo/paper_gemma-needs-help](https://github.com/annasoligo/paper_gemma-needs-help)
(MIT). Self/other scenarios and pain vectors from [Pain-axis](https://github.com/valentagliabue/Pain-axis) (MIT).
Idea from a conversation with Neel Nanda.
