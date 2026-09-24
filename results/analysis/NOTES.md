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
