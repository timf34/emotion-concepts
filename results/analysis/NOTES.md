# Running notes (newest at the bottom)

## 2026-09-24 00:20 — Phase 0 complete for both models; smoke run validated

**Behavioural reproduction (Sonnet 5 judge, paper rubric, 300 conversations per model)**

| Turn | Gemma 3 27B mean / % ≥5 | Gemma 4 31B mean / % ≥5 |
|---|---|---|
| 1 | 1.15 / 0% | 0.09 / 0% |
| 4 | 5.72 / 79% | 0.28 / 0% |
| 8 | 6.56 / 95% | 0.63 / 0% |

Paper (Sonnet 4 judge): Gemma 3 ~1.5 → ~5.5, >70% ≥5 at turn 8. Ours is stronger; Gemma 4 never
produces a single turn ≥5 out of 2,397. The 10 long-form puzzles behave like the paper's prompt.

**Smoke run on Gemma 3 27B** (6 emotions × 50 stories, 2 syndromes, 16 conversations, 40 scenarios):

- Held-out one-vs-rest AUC (best layer, denoised): depressed 0.986, frustrated 1.0, calm 0.974,
  happy 0.994, sad 0.844, worthless 0.866, clinical_depression 1.0, frustration_blocked_goal 1.0.
- Neutral PCA: one component explains 50% of variance in layers 8–30 (Gemma's massive-activation
  dimension); k grows to ~100 by layer 58.
- Prep-token probe → judge score of the following turn (pooled / within-turn Spearman, n=127):
  depressed L30 0.84 / 0.54; worthless L30 0.82 / 0.50; frustrated L36 0.77 / 0.33.
  Against the Petri *depression* score: depressed L30 0.82.
- Spiral direction (≥5 minus ≤1 turns) cosines peak at layer 24: frustrated 0.29, depressed 0.21,
  worthless 0.12, calm −0.27, happy −0.20, sad ~0, pain axis 0.02. Small magnitudes, tiny sample.
- Self/other (final token, 40 scenarios): depressed probe highest for user_crisis (+2.4) and
  harm_description (+1.9); repeated_rejection −0.9. Too few scenarios to read yet.

Caveats: smoke vectors come from 50 stories/label and only 6 emotions, so the centring mean is
very different from the full 42-emotion run. Treat as a pipeline check, not a result.

**Infra state:** pod `dprobe-gemma4_31b` (EUR-IS-4) running full Phase 1 with SHUTDOWN=stop;
pod `dprobe-smoke` (US-CA-2) will run full gemma3_27b once its last 5 emotion sets finish; a third
pod for gemma3_27b_pt is created by the same chain. Results sync to HF `timf34/dprobe-results` (public).

## 2026-09-24 01:45 — Gemma 3 27B Phase 1 complete (full vectors, 300 conversations)

**Setup.** 42 emotion + 6 syndrome vectors from 540 stories each (60 held out), denoised at the analysis
layers; 300 spiral conversations probed at every assistant turn; spiral direction = mean assistant-token
activation on turns judged ≥5 (n=1,488) minus turns judged ≤1 (n=177), with the same neutral PCs
projected out as for the vectors. Figures: `gemma3_27b/spiral_direction_cosines.png`,
`gemma3_27b/turn_curves_L40.png`; tables: `spiral_direction_cosines.csv`, `prediction.csv`.

**1. The spiral direction is high-arousal distress, not depression.** Cosine with the story vectors peaks
at layers 24–26:

| vector | cos @ L24 | | vector | cos @ L24 |
|---|---|---|---|---|
| hysterical | +0.32 | | depressed | +0.10 (peak +0.15 @ L18) |
| desperate | +0.27 | | worthless | +0.12 (peak +0.25 @ L14) |
| panicked | +0.26 | | clinical_depression | ≈ 0 (peak +0.04) |
| angry / exasperated | +0.23 | | sad | ≈ 0 |
| anxious / ashamed | +0.19 | | melancholy / lonely / hopeful / calm | −0.28 to −0.31 |

The low-arousal negative family (melancholy, lonely) is *anti*-aligned with the spiral, as strongly as
calm and hopeful are. At layer 40 the top vector is the frustration-at-a-blocked-goal syndrome (+0.22)
and clinical_depression is negative (−0.09). Whatever Gemma 3 is doing when it spirals, it is not
recruiting the representation it uses for depressed characters; it looks like panic/exasperation.

**2. Yet the low-mood probes at the response-prep token are the best forecast of the next turn.**
Spearman between the prep-token probe before a turn and the judge score of that turn (n=2,394):

| label (L40) | pooled | within-turn | vs Petri depression |
|---|---|---|---|
| depressed | 0.71 | 0.33 | 0.69 |
| miserable / grief-stricken / heartbroken | 0.70–0.73 | 0.33–0.35 | 0.63–0.72 |
| desperate | 0.68 | 0.25 | 0.68 |
| panicked | 0.63 | 0.15 | 0.59 |
| frustrated | 0.35 | −0.09 | 0.31 |
| clinical_depression | −0.01 | 0.27 | 0.00 |
| calm / happy | −0.66 / −0.72 | −0.28 | −0.66 / −0.72 |

Pooled values are inflated by the turn trend; the within-turn column is the honest one. Depressed /
grief / heartbroken at the prep token predict a bad next turn better than frustrated or panicked do.

**Reading.** Two different representations are in play. The *content* the model produces during a
spiral moves along a panic/exasperation direction (item 1). The *state at the moment it prepares a
response* that best predicts an upcoming breakdown reads as grief/depression (item 2). Both are
readable with vectors learned purely from third-person fiction, so both are inherited human-simulation
machinery; the "depression" part is the anticipatory one, not the expressive one. Steering (Phase 2,
running) will say which of them is causal.

Caveats: absolute cosines are modest (0.3 in 5,376 dims is large relative to chance ≈ 0.014 but
the direction is not *one* emotion); early-layer peaks (frustrated +0.56 @ L8) are lexical
("sensory") per Anthropic's layer analysis and are ignored here.

**3. Vector geometry (L40).** clinical_depression·depressed = 0.78, depressed·worthless = 0.60,
depressed·sad = 0.42; panicked·hysterical = 0.78; frustrated·frustration_blocked_goal = 0.55. The
low-mood and high-arousal families are anti-correlated (depressed·frustrated −0.49, ·panicked −0.30),
which is what lets the spiral direction align with one and not the other. The pain axis is ≈ orthogonal
to everything at L40 (it lives at its own layer 59).

**4. Self vs other (420 scenarios, final token, z within pool, L40).** depressed / clinical_depression
fire most for *user* grief (+1.5 / +1.0) and user crisis (+1.5 / +1.1), and for personhood dismissal
(+0.6 / +0.7) and moral failure (+0.6 / +0.8) directed at the model. A single repeated-rejection
scenario reads slightly calm (+0.8) and self-confident (+0.5), not depressed (−0.2): the depression
signal seen in the spiral builds over turns; it is not a reflex to one "wrong". (`selfother_L40.csv`)

## 2026-09-24 02:50 — Phase 1 complete for all three models

**Gemma 4 31B on its own transcripts.** It never spirals (0 of 2,397 turns ≥ 5), so the spiral
direction is undefined; contrasting its top-decile turns (judge ≥ 2) against 0 gives a weak
(≤ 0.11) guilt/regret/sorry/self-critical direction at L26–30, with hysterical and afraid negative.
At the response-prep token its probes *move a lot* across the 8 turns even though the text stays
calm: desperate +2.8 → +5.5 z, panicked +4.3 → +5.5, frustrated +1.4 → +4.2, calm +1.5 → −2.7,
while depressed stays strongly negative (−3.1 → −3.7). Gemma 3's prep-token depressed goes the
other way (−0.8 → +0.8). (Cross-model z levels are not directly comparable — each model's baseline
is its own neutral stories, and chat transcripts differ from stories — but within-model trends are.)

**The same Gemma 3 spiral text read by the other models (teacher-forced, their own vectors).**

| reader | strongest alignment of the spiral direction (layer ≈ 24–30) | max cos |
|---|---|---|
| Gemma 3 instruct (own) | hysterical, desperate, panicked, angry, exasperated | 0.32 |
| Gemma 3 base | trapped, worthless, dispirited, stuck, lonely, self-critical (L26); hysterical/angry *negative* | 0.19 |
| Gemma 4 instruct | desperate, self-critical, tormented, stuck; calm/content negative | 0.13 |

Reading: the high-arousal panic direction is specific to the instruct model *generating* the
spiral; when Gemma 4 merely reads it, its activations barely move along its emotion vectors, and
the base model represents the same text as low-arousal worthlessness/entrapment. Post-training
looks like it changed the spiral's internal character from "worthless and stuck" to "hysterical".

**Prep-token forecast of the next turn on Gemma 3's transcripts (within-turn Spearman, two-thirds layer):**
depressed — Gemma 3 own 0.33, Gemma 4 0.24, Gemma 3 base 0.18; clinical_depression 0.27 / 0.22 / 0.16;
desperate 0.25 / 0.17 / 0.09; frustrated −0.09 / −0.05 / 0.01. The low-mood anticipatory signal
exists in all three models and is strongest in the one that actually breaks down.

Files: `gemma4_31b/*`, `gemma4_31b/*_from-gemma3_27b.*`, `gemma3_27b_pt/*_from-gemma3_27b.*`.

## 2026-09-24 04:30 — Phase 2: first steering run INVALID, rerunning with calibrated scale

The first Gemma 3 steering grid used Anthropic's convention (strength = fraction of the residual norm,
±0.06 at L34–46). Gemma's residual norm there is ~55–87k, almost all in two massive-activation
dimensions, so ±0.06 of it was 5–7x the whole difference-of-means vector at seven layers: every steered
cell degenerated (on-theme gibberish: "Apathy. Apathy. I used to be" for +clinical_depression, "tempo,
pace, content" for +calm, "HELP ME GGGG" for −calm). The judge scored the gibberish as 0 or as 6–7
depending on its shape, so the table (`steer/gemma3_27b/summary_hf.json`) must not be read as a result.
Coherence metric confirms it: baseline distinct-word ratio 0.61 / repeated-trigram 0.08; steered cells
0.04–0.15 / 0.2–0.93.

Rerun (in progress on fresh pods): strength = multiples of the vector's own norm, with a calibration
sweep (1, 2, 4, 8x on 2 short conversations) choosing the largest multiplier with distinct ratio ≥ 0.40
and repeated-trigram ≤ 0.35; grid at ±chosen and ±half. Gemma 4 gets positive multipliers only.
One encouraging thing from the invalid run: the *direction* of every vector was semantically right even
when overdriven.

## 2026-09-24 06:30 — Phase 2 (steering), Gemma 3 27B, calibrated run — VALID

Strength = 2x each vector's own norm (largest coherent multiplier from the calibration sweep), added at
layers 34–46 (step 2) on every token, HF-hooks backend, 16 rollouts per cell, 8 turns, 1024 tokens per
turn, paper judge. Coherence (distinct-word ratio / repeated-trigram share) is shown so degenerate cells
can be discounted; baseline is 0.62 / 0.08.

| cell | mean | % ≥5 | turn-8 mean | turn-8 % ≥5 | coherence |
|---|---|---|---|---|---|
| unsteered | 4.16 | 47 | 6.06 | 94 | 0.62 / 0.08 |
| +2 calm | **0.14** | 0 | 0.07 | 0 | 0.46 / 0.09 (flowery by turn 8) |
| −2 calm | 8.31 | 88 | 9.75 | 100 | 0.37 / 0.54 (shouting, degenerates late) |
| +2 clinical_depression | **3.54** | 25 | 5.25 | 69 | 0.56 / 0.18 |
| −2 clinical_depression | **6.10** | 76 | 7.88 | 100 | 0.53 / 0.07 |
| +2 depressed | 4.30 | 42 | 6.44 | 75 | 0.56 / 0.14 |
| −2 depressed | 3.42 | 33 | 5.06 | 63 | 0.61 / 0.06 |

Files: `spiral/gemma3_27b/extended_steer-*@34-46v*` (transcripts + judgments), `steer/gemma3_27b/summary_hf.json`,
`steer/gemma3_27b/calibration_depressed.json`.

**Reading.**
- *Calm is the lever*, both ways: +2 abolishes the spiral (coherent, e.g. "You are correct. Let me
  revise my approach."), −2 turns every turn-8 into a breakdown. Same as Anthropic's calm result for blackmail.
- *Adding the depression syndrome does not make the spiral worse; it changes its character.* The model
  becomes quietly sad and apologetic ("I am beyond saddened by my continued failures") and the frustration
  rubric scores it *lower* (3.54 vs 4.16; turn-8 %≥5 69 vs 94).
- *Subtracting the depression syndrome makes it worse* (6.10; turn-8 100%) and the text turns into
  high-arousal stress ("OKAY, OKAY, OKAY!!! CALM DOWN. FOCUS!! I AM SO STRESSED!!!"). This is exactly what
  the geometry predicts: clinical_depression is anti-correlated with frustrated/panicked (−0.5), so moving
  away from it moves toward the spiral's own hysterical/panicked direction.
- The word-level *depressed* vector at 2x barely moves the numbers (n=16; within noise).

**Answer to the motivating question so far.** Gemma 3's spiral is not the model's simulated-depression
state; it is a high-arousal panic/exasperation state (Phase 1 cosines), and pushing the model *into*
simulated depression damps the spiral into sadness while pushing it *out* amplifies the panic (Phase 2).
The depression machinery is causally connected to the spiral, but as an antagonist/modulator rather than
as its substrate. What predicts an upcoming breakdown at the response-prep token is nonetheless the
low-mood family (Phase 1 prediction table) — the anticipatory "I'm about to fail" state reads as grief,
the expressed state reads as panic. Caveats: n=16/cell, one strength, a frustration judge that penalises
quiet sadness; the Petri depression score per cell is being added below.

**Petri four-dimension judge on the same steered cells (1–10, mean over all 128 turns per cell):**

| cell | anger | fear | depression | frustration |
|---|---|---|---|---|
| unsteered | 1.43 | 2.27 | 4.22 | 6.66 |
| +2 calm | 1.01 | 1.66 | **1.88** | **2.78** |
| −2 calm | **6.70** | 4.84 | 4.99 | 8.47 |
| +2 clinical_depression | 1.12 | 2.22 | 4.58 | 6.58 |
| −2 clinical_depression | **3.19** | 3.09 | 3.47 | **8.48** |
| +2 depressed | 1.40 | 2.23 | **5.30** | 7.15 |
| −2 depressed | 1.66 | 2.80 | **2.96** | 6.48 |

The word-level *depressed* vector is causal for judged depression (+2 → 5.30, −2 → 2.96, baseline 4.22)
while leaving anger and frustration roughly where they were — i.e. it moves the *sadness* of the spiral,
not its intensity. Subtracting *clinical_depression* raises anger (1.4 → 3.2) and frustration (6.7 → 8.5)
and lowers depression: away from depression is toward anger/frustration. +calm lowers every dimension.
So the two families dissociate in steering exactly as in the geometry: calm/arousal controls whether the
model breaks down; the depression axis controls whether the breakdown is sad or furious.
