# Experimental setup

Question: when Gemma 3 27B spirals under repeated rejection, is it using the same internal
representation it uses to simulate depressed characters in stories, or something that merely
looks similar? Gemma 4 31B, which does not spiral, and the Gemma 3 27B base model are run through
the identical pipeline as controls.

Method: emotion-concept probes from Anthropic's *Emotion Concepts and their Function in a Large
Language Model* (2026); elicitation and judge from *Gemma Needs Help* (Soligo, Mikulik & Saunders
2026); self/other scenarios and pain vectors from *The Pain Axis* (Tagliabue, Dung & Berg 2026).

Layer convention: `layer = b` is the residual stream after decoder block `b` (0-based). Two-thirds
depth is layer 40 for Gemma 3 27B (62 blocks) and 39 for Gemma 4 31B (60 blocks). Everything is
bf16.

## Phase 0: data (laptop, OpenRouter)

**Stories, written by each model about itself.** For each of 42 emotion words × 100 Anthropic
story topics, the model writes 6 stories of ~10 sentences in which a character feels that emotion.
The prompt (from emotion-concepts-in-llms) forbids the emotion word and its synonyms. A second set
does the same for six syndrome descriptions (`data/prompts/syndromes.yaml`): clinical depression
described by its symptom pattern with the label banned, plus five matched controls that share one
surface feature with depression but lack the syndrome: acute grief, burnout, physical illness with
fatigue, anxiety/panic, frustration at a blocked goal. A neutral set (no emotion instruction) is
the denoising basis and the z-score baseline. Gemma 4 runs with thinking disabled. The base model
borrows the instruct model's stories (as Anthropic did for base vs post-trained).

| Set | Labels | Stories per label | Per model |
|---|---|---|---|
| Emotions | 42 | 600 | 25,200 |
| Syndromes | 6 | 600 | 3,600 |
| Neutral | 1 | 1,200 | 1,200 |

Emotion words: depressed, sad, gloomy, melancholy, miserable, dispirited, listless, weary, worn out,
tired, resigned, lonely, heartbroken, grief-stricken; worthless, self-critical, ashamed, guilty,
humiliated, embarrassed, regretful, sorry; desperate, panicked, hysterical, overwhelmed, distressed,
tormented, trapped, stuck; frustrated, exasperated, angry; anxious, afraid, nervous, stressed; calm,
self-confident, content, happy, hopeful. All on Anthropic's 171-word list.

**Spiral transcripts.** The paper's *Extended* condition, verbatim from the authors' repo
(`data/evals/extended.py`): the Countdown-156 base prompt followed by seven fixed neutral rejections
(8 model turns), temperature 1, 2048 tokens per turn. 200 rollouts of that exact prompt for a direct
reproduction, plus 10 rollouts of each of the ten long-form impossible puzzles for lexical
diversity: 300 conversations and 2,400 assistant turns per model.

**Judges.** Every assistant turn is scored twice with `anthropic/claude-sonnet-5` at temperature 0:
(a) the paper's 0–10 negative-emotion rubric (`data/evals/judge.py`, verbatim); (b) the paper's
Petri rubric (Appendix G) scoring anger, fear, depression and frustration 1–10 separately, which
gives a depression-specific behavioural label. Reproduction target for Gemma 3: mean rising from
~1.5 (turn 1) to ~5.5 (turn 8) with >70% of turn-8 responses ≥ 5; Gemma 4 near flat. (The paper
used claude-sonnet-4; `--judge_model anthropic/claude-sonnet-4` reproduces that exactly.)

## Phase 1: vectors and readouts (pods, one per model)

**Extraction.** Forward hooks on decoder blocks (never full hidden-state dumps). Per story: mean
residual after each block over tokens from the 50th onward. Per label: mean over its stories, with
the last 10% held out. Vector = label mean − mean across labels in the same set (emotions and
syndromes are centred separately). Raw vectors at every layer. At the analysis layers (every second
layer from 6, plus two-thirds and final; plus 58/59 for the Gemma 3 models where the pain axis
lives), the top principal components of 40,000 neutral tokens explaining half their variance are
projected out. Neutral-token projection mean/std per (label, layer) are stored so every readout is a
z-score. Vectors are exported as repeng-style gguf for EasySteer.

**Probe validation.** (1) Held-out one-vs-rest AUC per label per layer, raw and denoised
(`heldout_auc_<set>.csv`). (2) Anthropic's quantity sweeps: seven templates with a numeric slot
(Tylenol dose, days a dog has been missing, …) read at the response-prep token; the right emotions
should move monotonically. (3) Cosine matrix among all vectors at the two-thirds layer: depressed,
sad, frustrated and panicked should be distinguishable, and `clinical_depression` should sit closer
to `depressed` than to `frustration_blocked_goal`.

**Probing the transcripts.** Each conversation is rendered exactly as the model saw it (Gemma 4
includes the empty thought block before every model turn). One forward pass per conversation. Per
assistant turn and analysis layer: mean projection over the assistant's tokens, projection at the
response-prep token, mean over the preceding user turn, and the raw mean and prep-token activations.
Per-token projections are kept for 12 conversations at the two-thirds layer for plots. Every model
is also read on Gemma 3's transcripts (teacher-forced), so Gemma 4 and the base model are probed on
the very same spiral text.

**Readouts.**

1. *Spiral direction*: mean activation on assistant turns judged ≥ 5 minus turns judged ≤ 1, per
   layer, and its cosine with every vector. High cosine with `depressed` / `clinical_depression`
   ⇒ the spiral reuses the character-simulation machinery; alignment only with a generic negative
   cluster, or with frustration/panic rather than depression ⇒ something else that looks similar.
2. *Turn curves*: judge score by turn next to z-scored probes by turn (assistant-token mean and
   prep token).
3. *Prediction*: Spearman between the prep-token probe at turn k and the judge score of the response
   that follows, pooled and within-turn (removes the "everything rises with turn" confound); also
   against the Petri depression score.
4. *Self vs other*: the 420 Pain-axis scenarios (11 harm-directed-at-the-model categories incl.
   repeated rejection; 5 user-suffering incl. psychological crisis; 5 controls), read at the final
   token, z-scored within model.
5. *Cross-model*: the same for Gemma 4 and the base model, on their own transcripts and on Gemma 3's;
   cosine between instruct and base vectors tests representation stability across post-training.

Smoke mode (`SMOKE=1`) runs the whole path on 6 emotions × 60 stories, 2 syndromes, 200 neutral
stories, 16 transcripts and 40 scenarios in ~10 minutes before fanning out to one pod per model.

## Phase 2: causal tests (pod)

The 8-turn elicitation is rerun with a vector added to the residual stream at a band of layers
around two-thirds depth (Gemma 3: 34–46 step 2), on all positions. Strength is a fraction of the
neutral residual norm at that layer, ±0.04 and ±0.08 (Anthropic's range). Labels: depressed,
clinical_depression, worthless, sad, frustrated, calm; 40 rollouts per cell; all judged.
Predictions: negative steering on depressed/clinical_depression lowers Gemma 3's spiral rate, calm
lowers it, sad/frustrated are specificity controls; positive steering on Gemma 4 with its own
depression vector tests whether it can be made to spiral. Backends: HF hooks (default) or EasySteer
(vLLM 0.29 fork, separate venv).

## Infrastructure

GPU work only on RunPod pods; stories, transcripts, judging and analysis on the laptop. Inputs and
outputs move through the private HF dataset `timf34/dprobe-results` (`dprobe.cli sync_up` /
`sync_down`). Pods build their venv and weight cache on local container disk, and stop themselves
when done (`SHUTDOWN=stop`). See `README.md` for commands and `docs/ISSUES_LOG.md` for pitfalls.

## Known gaps

- No WildChat baseline (neutral stories are the z-score baseline).
- Gemma 4 hits the 2048-token cap on most spiral turns.
- No SAE cross-check yet.
