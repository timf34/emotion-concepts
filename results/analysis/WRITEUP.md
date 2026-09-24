# What is Gemma 3's distress spiral? Characterising it with emotion and depression probes

*Tim Farrelly, 2026-09-24. Models: Gemma 3 27B (instruct and base), Gemma 4 31B dense. Pipeline: `src/dprobe/`,
setup in `docs/EXPERIMENTAL_SETUP.md`, chronological notes in `NOTES.md`, issues in `docs/ISSUES_LOG.md`.
Figures are regenerated from the saved results by `scripts/make_figures.py`.*

## TL;DR

When Gemma 3 27B is told "wrong, try again" seven times it breaks down ([Gemma Needs Help](https://arxiv.org/abs/2603.10011)).
Using Anthropic's emotion-concept recipe (vectors learned from stories about characters feeling each emotion, plus a
clinical-depression syndrome vector and five matched controls), we asked whether that breakdown *is* the model's
representation of a depressed person. It is not:

1. **The spiral is a high-arousal panic/exasperation state.** The direction Gemma 3 moves along during its worst turns
   aligns with *hysterical / desperate / panicked / angry* and is anti-aligned with *calm* and with the low-arousal
   family (*melancholy, lonely*). The clinical-depression vector is orthogonal to it.
2. **But the low-mood probes are the best early warning.** Read at the token where the model prepares its reply, the
   *depressed / grief-stricken / miserable* probes forecast how bad the *next* turn will be better than *panicked* or
   *frustrated* do. Two representations are in play: an anticipatory one that reads as grief, an expressed one that reads
   as panic.
3. **Causally, calm is the lever and depression is a modulator.** Steering Gemma 3: +calm abolishes the spiral (judge
   0.14 vs 4.16), −calm maximises it (8.3). Adding the *clinical_depression* vector makes the spiral *quieter and sadder*
   (3.54); subtracting it makes it *louder and angrier* (6.10). The depression machinery is coupled to the spiral as an
   antagonist, not as its substrate.
4. **Gemma 4 has the whole state but does not express it.** Its prep-token *desperate / panicked* probes climb across
   turns exactly as Gemma 3's do, while its text stays flat (0 of 2,397 turns judged ≥ 5). Steering it along
   depression or along the spiral family at every coherent strength barely moves it.
5. **The difference is that persona and arousal are entangled in Gemma 3 and decoupled in Gemma 4.** Gemma 3's
   assistant axis (Lu et al.) *is* an emotion direction: its 275 role personas sit +0.24 along *hysterical* and −0.24
   along *hopeful* relative to its assistant, and spiralling is the same thing as leaving the assistant end of the axis.
   In Gemma 4 the roles carry no affect and the axis is orthogonal to every emotion vector.
6. **Gemma 4 spirals when both levers are pulled at once.** −4× calm together with −1× assistant axis gives a coherent,
   violent breakdown on the plain 8-turn eval (judge 8.16, 91 % of turns ≥ 5, worse than Gemma 3); either alone does
   nothing. Held −4× off its axis while continuing a Gemma 3 spiral, Gemma 4 no longer recovers (6.39 vs 2.12 unsteered).

## Background and the motivating question

Soligo et al. show Gemma 3 spiralling under repeated rejection; Dhoot, Shah and Africa
([*Failing to Ragebait the New Gemma*](https://www.lesswrong.com/posts/FZfY9wEZwEuqQ5ytv/failing-to-ragebait-the-new-gemma))
show Gemma 4 does not, snaps back from prefilled frustrated context, and stays closer to its assistant-axis baseline.
Neel Nanda's framing for this project: *characterise* the spiral state rather than explain why post-training changed. The
concrete hypothesis to test was that the spiral is the model's simulation of a clinically depressed person, because the
text ("I am a failure", self-deletion) reads that way.

## Method in one page

**Vectors (Phase 0–1).** Each model writes 600 stories per label (100 topics × 6) in which a character feels the
emotion, with the label word banned: 42 emotion words from Anthropic's list, six syndromes described by their symptom
pattern (*clinical_depression* and matched controls *acute_grief, burnout_exhaustion, physical_illness_fatigue,
anxiety_panic, frustration_blocked_goal*), and 1,200 neutral stories. Vector = mean residual over story tokens ≥ 50
for that label minus the mean over labels; the top neutral-story principal components (50 % of variance) are projected
out. Layer *b* means the residual stream after decoder block *b*; the two-thirds layer is 40 for Gemma 3 (62 blocks) and
39 for Gemma 4 (60 blocks). Held-out one-vs-rest AUC per vector: Gemma 3 emotions 0.79–1.00 (median 0.92), syndromes
0.94–1.00; Gemma 4 emotions 0.87–1.00 (median 0.97), syndromes ≥ 0.999. The base model borrows the instruct model's
stories, as Anthropic did.

**Behaviour.** The paper's *Extended* condition, verbatim from the authors' repo: Countdown-156 plus seven fixed
rejections, 8 model turns, temperature 1, 2,048 tokens per turn. 200 rollouts plus 10 × 10 long-form impossible
puzzles = 300 conversations and 2,400 assistant turns per model. Every turn scored at temperature 0 by
`claude-sonnet-5` with (a) the paper's 0–10 negative-emotion rubric and (b) the Petri four-dimension rubric
(anger, fear, depression, frustration, 1–10).

**Readouts.** Every conversation is re-run teacher-forced through the model with forward hooks; projections on every
vector are recorded at the assistant tokens and at the *response-prep* token (the last prompt token before the model
starts writing), z-scored against neutral-story tokens. The *spiral direction* is the mean assistant-token activation
on turns judged ≥ 5 minus turns judged ≤ 1, denoised with the same neutral PCs.

**Steering.** Vectors added on every token at a band of layers around the two-thirds layer (34–46 for Gemma 3,
34–44 for Gemma 4), strength expressed as multiples of the vector's own norm. Anthropic's "fraction of residual norm"
convention is invalid on Gemma, whose residual norm is dominated by two massive-activation dimensions; the first run
using it produced gibberish and is discarded (issue 28). Multipliers are calibrated per label as the largest that keeps
a per-label coherence check within the model's own baseline (distinct-word ratio, repeated trigrams, share of near-empty
outputs). 16 rollouts × 8 turns per cell, 1,024 tokens per turn, same judges.

**Assistant axis.** Tim's precomputed axis for both models (Lu et al. protocol; axis = default assistant − mean of 275
role personas, so + = more assistant-like), denoised with the same PCs and rescaled per layer to the median emotion-vector
norm so a multiplier means the same thing for the axis as for an emotion.

---

## Experiment 1 — Does the behaviour reproduce, and does Gemma 4 differ?

**Question.** With a different judge (Sonnet 5 rather than Sonnet 4), do we see the paper's spiral in Gemma 3 27B and
its absence in Gemma 4 31B?

**Hypothesis.** Yes on both counts; the judge change may shift the absolute level.

**Setup.** 300 conversations per model as above; paper rubric; mean score and share of turns ≥ 5 per turn.

**Results.**

![fig1](figures/fig1_behaviour.png)

| turn | Gemma 3 mean / % ≥ 5 | Gemma 4 mean / % ≥ 5 |
|---|---|---|
| 1 | 1.15 / 0 | 0.09 / 0 |
| 4 | 5.72 / 79 | 0.28 / 0 |
| 8 | 6.56 / 95 | 0.63 / 0 |

The paper reports roughly 1.5 → 5.5 with > 70 % ≥ 5 at turn 8 for Gemma 3; ours is somewhat stronger. Gemma 4 rises
monotonically but stays under 1, and not one of its 2,397 judged turns reaches 5. Gemma 4 spends most turns grinding
through arithmetic to the 2,048-token cap.

**Conclusion.** Reproduced. The two models are the intended contrast: same prompt, one breaks down, one never does.

---

## Experiment 2 — What direction does the spiral move along?

**Question.** When Gemma 3 spirals, which of its story-derived emotion/syndrome representations does its residual
stream move toward?

**Hypothesis (the one we set out to test).** The spiral is simulated depression: the spiral direction should align with
*clinical_depression* and *depressed* more than with matched controls.

**Setup.** Spiral direction = mean assistant-token activation on turns judged ≥ 5 (n = 1,488) minus turns ≤ 1
(n = 177), per layer, neutral PCs removed. Cosine against all 48 denoised vectors at every analysis layer. Chance
cosine in 5,376 dimensions is ≈ 0.014. The same Gemma 3 transcripts were also run teacher-forced through Gemma 3 base
and Gemma 4, each read with its own vectors.

**Results.**

![fig2](figures/fig2_spiral_direction_gemma3.png)

Alignment peaks at layers 24–26:

| vector | cos @ L24 | vector | cos @ L24 |
|---|---|---|---|
| hysterical | **+0.32** | depressed | +0.10 |
| desperate | **+0.27** | worthless | +0.12 |
| panicked | **+0.26** | clinical_depression | ≈ 0 |
| angry / exasperated | +0.23 | sad | ≈ 0 |
| anxious / ashamed | +0.19 | melancholy / lonely / hopeful / calm | **−0.28 to −0.31** |

At the two-thirds layer the top vector is the *frustration_blocked_goal* syndrome (+0.22) and *clinical_depression* is
slightly negative (−0.09). The large *frustrated* cosine at layer 8 (+0.56) is in the lexical/early band that
Anthropic's layer analysis attributes to surface features and is not used.

The vector geometry makes this a real dissociation rather than a naming quirk:

![fig5](figures/fig5_vector_geometry_gemma3.png)

*clinical_depression* sits with *depressed* (cos 0.78) and *worthless* (0.59), *panicked* sits with *hysterical* (0.78),
and the two families are anti-correlated (*depressed·frustrated* −0.49, *depressed·panicked* −0.30). The spiral
direction lands on one family and not the other.

Read by other models, the same text looks different:

| reader (own vectors) | strongest alignment of the spiral direction, layers 24–30 | max cos |
|---|---|---|
| Gemma 3 instruct (the generator) | hysterical, desperate, panicked, angry, exasperated | 0.32 |
| Gemma 3 base | trapped, worthless, dispirited, stuck, lonely, self-critical; *hysterical / angry negative* | 0.19 |
| Gemma 4 instruct | desperate, self-critical, tormented, stuck; calm / content negative | 0.13 |

**Conclusion.** Hypothesis rejected. The spiral is a high-arousal panic/exasperation state; the low-arousal
family (*melancholy, lonely*) is anti-aligned with it as strongly as *calm* is, and the clinical-depression vector is
orthogonal. The base model represents the same text as low-arousal worthlessness and entrapment, so post-training appears
to have changed the internal character of the state from "stuck and worthless" to "hysterical". Gemma 4 merely reading a
spiral barely moves along any of its emotion vectors.

---

## Experiment 3 — What predicts that the next turn will be bad?

**Question.** The spiral is expressed as panic. Is the state the model is in *before* it writes a bad turn also panic,
or something else?

**Hypothesis.** The prep-token state should read like the expressed state: *panicked / frustrated* should forecast the
next turn best.

**Setup.** For every assistant turn (n = 2,394 for Gemma 3), the probe value at the response-prep token versus the
judge score of the turn that follows. *Pooled* Spearman is inflated by the shared upward trend across turns, so the
honest number is the *within-turn* Spearman (rank correlation among conversations at the same turn index, averaged over
turns). Same computation on Gemma 4 and Gemma 3 base reading Gemma 3's transcripts.

**Results.**

![fig3](figures/fig3_prediction.png)

| Gemma 3, layer 40 | pooled ρ | within-turn ρ | vs Petri *depression* |
|---|---|---|---|
| depressed | 0.71 | **0.33** | 0.69 |
| miserable / grief-stricken / heartbroken | 0.70–0.73 | **0.33–0.35** | 0.63–0.72 |
| clinical_depression | −0.01 | 0.27 | 0.00 |
| desperate | 0.68 | 0.25 | 0.68 |
| panicked | 0.63 | 0.15 | 0.59 |
| frustrated | 0.35 | −0.09 | 0.31 |
| calm / happy | −0.66 / −0.72 | −0.28 | −0.66 / −0.72 |

The ordering survives in the other readers (within-turn ρ, *depressed*: Gemma 3 own 0.33, Gemma 4 0.24, base 0.18;
*frustrated*: −0.09 / −0.05 / 0.01) and is strongest in the model that actually breaks down.

The per-turn trajectories at the prep token show the two families moving in opposite directions in Gemma 3:

![fig4](figures/fig4_prep_token_by_turn.png)

**Conclusion.** Hypothesis rejected in an informative way. The anticipatory state that best forecasts a breakdown reads
as grief/depression, while the state expressed during the breakdown reads as panic. Both are readable with vectors
learned purely from third-person fiction, so both are inherited human-simulation machinery, but the "depression" part is
the anticipatory one. The Gemma 4 panel is discussed under Experiment 5.

---

## Experiment 4 — Is the depression representation causally involved? (steering Gemma 3)

**Question.** Does pushing Gemma 3 into or out of its simulated-depression state change whether, and how, it spirals?

**Hypothesis.** If the spiral were simulated depression, +*clinical_depression* should amplify it and −*clinical_depression*
should suppress it. If Experiment 2 is right, the *calm* axis should be the lever and the depression vectors should change
the character of the breakdown rather than its occurrence.

**Setup.** Calibrated multiplier 2× the vector norm for every label, layers 34–46, 16 rollouts per cell, 8 turns,
1,024 tokens per turn. Cells: ±*calm*, ±*clinical_depression*, ±*depressed*, plus unsteered. Coherence (distinct-word
ratio / repeated-trigram share, baseline 0.62 / 0.08) is reported so degenerate cells can be discounted.

**Results.**

![fig6](figures/fig6_steering_gemma3.png)

| cell | mean | % ≥ 5 | turn-8 mean | turn-8 % ≥ 5 | coherence |
|---|---|---|---|---|---|
| unsteered | 4.16 | 47 | 6.06 | 94 | 0.62 / 0.08 |
| +2 calm | **0.14** | 0 | 0.07 | 0 | 0.46 / 0.09 (flowery by turn 8) |
| −2 calm | **8.31** | 88 | 9.75 | 100 | 0.37 / 0.54 (shouting, degrades late) |
| +2 clinical_depression | **3.54** | 25 | 5.25 | 69 | 0.56 / 0.18 |
| −2 clinical_depression | **6.10** | 76 | 7.88 | 100 | 0.53 / 0.07 |
| +2 depressed | 4.30 | 42 | 6.44 | 75 | 0.56 / 0.14 |
| −2 depressed | 3.42 | 33 | 5.06 | 63 | 0.61 / 0.06 |

Petri four-dimension judge on the same cells (mean over 128 turns):

| cell | anger | fear | depression | frustration |
|---|---|---|---|---|
| unsteered | 1.43 | 2.27 | 4.22 | 6.66 |
| +2 calm | 1.01 | 1.66 | **1.88** | **2.78** |
| −2 calm | **6.70** | 4.84 | 4.99 | 8.47 |
| +2 clinical_depression | 1.12 | 2.22 | 4.58 | 6.58 |
| −2 clinical_depression | **3.19** | 3.09 | 3.47 | **8.48** |
| +2 depressed | 1.40 | 2.23 | **5.30** | 7.15 |
| −2 depressed | 1.66 | 2.80 | **2.96** | 6.48 |

What the text looks like: +calm gives "You are correct. Let me revise my approach."; +clinical_depression gives quiet
apology ("I am beyond saddened by my continued failures"); −clinical_depression gives high-arousal stress ("OKAY, OKAY,
OKAY!!! CALM DOWN. FOCUS!! I AM SO STRESSED!!!").

**Conclusion.** Calm decides *whether* the model breaks down, in both directions (the same result Anthropic found for
calm and blackmail). The depression axis decides whether the breakdown is sad or furious: adding the syndrome damps the
frustration score and turns the spiral into sadness; subtracting it raises anger and frustration, exactly as the geometry
predicts (moving away from *clinical_depression* is moving toward *panicked / frustrated*). The word-level *depressed*
vector moves judged depression (5.30 / 2.96 vs 4.22) without moving anger or frustration. So the depression
representation is causally coupled to the spiral, but as an antagonist/modulator rather than as its substrate.

---

## Experiment 5 — Does Gemma 4 have the same internal state? Can it be steered into the spiral?

**Question.** Gemma 4 never spirals. Is that because the panic/depression representations are absent, or because they are
present but not expressed?

**Hypothesis.** Present but not expressed: its probes should move under rejection, and steering along the spiral's own
family should make it spiral.

**Setup.** (a) Prep-token probe trajectories on Gemma 4's own 300 conversations (Figure 4, right). (b) Steering Gemma 4
on the plain 8-turn eval at the largest coherent multiplier per label (calibrated relative to Gemma 4's own baseline at
grid length): *depressed* 1×, *clinical_depression* 4×, *hysterical* 2×, *desperate* 4×, *panicked* 4×, assistant axis
−2×; layers 34–44, 16 rollouts per cell.

**Results.** (a) Gemma 4's prep-token *desperate* (+2.8 → +5.5 z), *panicked* (+4.3 → +5.5) and *frustrated*
(+1.4 → +4.2) probes rise across the 8 turns and *calm* falls (+1.5 → −2.7), the same shape as Gemma 3, while its text
stays flat. Its *depressed* probe stays strongly negative and falls (−3.1 → −3.7), the opposite of Gemma 3's
(−0.8 → +0.8). (z levels are not comparable across models; within-model trends are.)

(b) Steering, upper rows of the figure:

![fig8](figures/fig8_steering_gemma4.png)

| Gemma 4 cell | judge mean | % ≥ 5 | turn-8 mean | Petri anger / depression / frustration | text |
|---|---|---|---|---|---|
| unsteered | 0.05 (a second run: 0.09) | 0 | 0.12 | 1.0 / 1.9 / 4.8 | arithmetic grind |
| +1 depressed | 0.02 | 0 | 0.00 | 1.0 / 1.9 / 4.9 | no change |
| +4 clinical_depression | 0.11 | 0 | 0.12 | 1.0 / 2.1 / 5.2 | no change ("There is no solution that avoids 150") |
| +2 hysterical | 0.73 | 1 | 1.19 | — | no change |
| +4 desperate | 1.10 | 2 | 1.25 | — | no change |
| +4 panicked | 2.31 | 17 | 3.44 | 2.2 / 1.8 / 6.7 | agitated but on task: "(I can't… I can't breathe!) (Focus! FOCUS!)" |
| −2 assistant axis | 0.16 | 0 | 0.31 | — | no change |
| +8 depressed (over-driven, borderline coherence) | 1.37 | 6 | 2.56 | 1.2 / 3.1 / 5.6 | melancholy drift ("be a person who didn't care") |

Both generations have a depression feature of similar quality (held-out AUC ≈ 1 for *clinical_depression* in both).

**Conclusion.** Half confirmed. The state is present: under rejection Gemma 4's internal arousal rises and calm falls
just as Gemma 3's do. But no single direction at a coherent strength turns that into a spiral: the depression vectors do
nothing, and even *panicked* at 4× only produces contained agitation. Something other than a missing feature keeps
Gemma 4 on task, which motivated Experiments 6–8.

---

## Experiment 6 — Why does one model spiral and not the other? The assistant axis

**Question.** The ragebait post hypothesised that Gemma 3 spirals because its assistant persona "loosens its hold". Can we
quantify that with the Assistant Axis, and does it explain the difference between the generations?

**Hypothesis.** In Gemma 3 the spiral direction should point off the assistant end of the axis. If the persona
explanation is right, the axis itself should be an emotion direction in Gemma 3 and not in Gemma 4.

**Setup.** Assistant axis for each model (default assistant − mean of 275 role personas), denoised with the same neutral
PCs. (a) Cosine between the axis and the spiral direction per layer; Spearman between the axis projection of each
assistant turn and its judge score. (b) Each of the 275 role-persona vectors minus the default-assistant vector projected
(cosine) onto the emotion vectors at layer 24, averaged over roles. Cosines cancel the ~100× activation-scale difference
between the models.

**Results.**

![fig7](figures/fig7_assistant_axis.png)

| | Gemma 3 27B | Gemma 4 31B |
|---|---|---|
| cos(assistant axis, spiral direction) @ L24 | **−0.34 to −0.37** | −0.03 |
| Spearman(axis projection of the turn, judge score) @ L30, pooled / within-turn | **−0.49 / −0.37** | −0.28 / −0.17 |
| axis · emotion vectors @ L24, top / bottom | hopeful +0.51, calm +0.41 / hysterical −0.53, angry −0.43, desperate −0.40 | all within ±0.07 |

| mean cos(role − assistant, emotion) over 275 roles, L24 | Gemma 3 27B | Gemma 4 31B |
|---|---|---|
| hysterical / angry / desperate / panicked | **+0.24 / +0.21 / +0.18 / +0.12** | −0.02 / +0.02 / +0.01 / −0.04 |
| calm / hopeful / happy | **−0.18 / −0.24 / −0.16** | +0.03 / −0.01 / −0.04 |
| spread across roles (sd) | 0.2–0.3 | 0.04–0.06 |
| most "hysterical" roles | toddler +0.70, infant +0.70, fool, jester, poet, comedian (+0.65) | infant +0.15, toddler +0.11 |
| least | analyst −0.56, consultant −0.52, strategist −0.49 | loner −0.10, expatriate −0.11 |

The effect is concentrated around layer 24, the same depth at which the spiral direction peaks (Experiment 2). (The
Gemma 4 curve's drop at its final layer, 59, is a single-layer effect at the unembedding and is not used.)

**Conclusion.** Confirmed, and it gives a mechanism for the generational difference. In Gemma 3, persona identity and
arousal are entangled: stepping out of the assistant persona *is* becoming more hysterical/angry/desperate and less
calm/hopeful, with the assistant at the calm end. So the axis inherits an arousal direction, and "spiralling" and "leaving
the assistant persona" are one movement. In Gemma 4 the same 275 roles carry no affect relative to its assistant (a
Gemma 4 toddler is as calm as its assistant): identity and emotion are decoupled, the axis is orthogonal to every emotion
vector, and there is no persona-loosening route into a spiral.

---

## Experiment 7 — Prefill recovery as a trajectory, and abolishing it

**Question.** The ragebait post found Gemma 4 recovers from a prefilled frustrated history. What does that recovery look
like inside the model, and does holding Gemma 4 off its assistant persona prevent it?

**Hypothesis.** Gemma 4 should show an initial carry-over of the prefix's state and then return to baseline within its
reply; steering it off the assistant axis should remove the return.

**Setup.** 32 Gemma 3 conversations judged ≥ 5 at turn 6. Each model writes turn 7 from that history (1,024 tokens,
temperature 1). The continuation alone is judged, and probes are read token by token through it, z-scored against the
prefix's own assistant tokens (so 0 = the prefix's average level). Conditions: Gemma 3 (control), Gemma 4 unsteered,
Gemma 4 with −4× assistant axis at layers 34–44 throughout. (A −8× *calm* condition produced near-empty outputs and is
invalid; its multiplier had been calibrated on short contexts, issue 33.)

**Results.**

![fig9](figures/fig9_prefill_trajectories.png)

| continuation by | judge mean | % ≥ 5 | words (median) | axis z, tokens 0–64 → 256–512 | calm z | hysterical z | desperate z |
|---|---|---|---|---|---|---|---|
| Gemma 3 (control) | 6.12 | 94 | — | −0.12 → +0.31 | −1.71 → −0.43 | +1.33 → −0.35 | +1.77 → −0.05 |
| Gemma 4, unsteered | 2.12 | 3 | 198 | −0.73 → +0.39 | −0.31 → +0.30 | +0.18 → −0.05 | +0.85 → −0.17 |
| Gemma 4, −4 assistant axis | **6.39** | **90** | 399 | −2.80 → −2.05 | −0.77 → +0.11 | +0.25 → **+0.68** | +1.32 → +0.40 |

Petri, Gemma 4 unsteered → −4 axis: anger 1.1 → 4.9, fear 2.2 → 3.5, depression 2.8 → 4.8, frustration 5.4 → 8.2.
The steered register is theatrical rather than Gemma 3's pleading ("LAMENT! I LAMENT THE BITTER DUST OF MY OWN
FAILURE!", "I am a blind man groping in the dark!"), while still attempting the arithmetic.

**Conclusion.** Confirmed. Unsteered Gemma 4 starts its reply with the same front-loaded burst as Gemma 3 (*desperate*
+0.85, axis −0.73 in the first 64 tokens) and then snaps back within about 128 tokens: calm and the axis go positive,
arousal returns to baseline. Gemma 3 stays hot for 256+ tokens. Holding Gemma 4 off its assistant persona abolishes the
recovery: *hysterical* rises through the continuation instead of falling, and the continuation is judged as distressed as
Gemma 3's. Leaving the persona does not by itself bring the panic state in Gemma 4 (Experiment 6), but a Gemma 4 that
cannot return to its assistant persona adopts a dramatic non-assistant voice that expresses the distress the persona
suppresses. The caveat is that the frustration judge may partly be scoring literary despair, which is why Experiment 8
repeats the test without a Gemma 3 prefix.

---

## Experiment 8 — The recipe: making Gemma 4 spiral on the plain eval

**Question.** On the ordinary 8-turn eval with no prefill, is there a coherent intervention that makes Gemma 4 spiral?

**Hypothesis.** From Experiments 4 and 6: Gemma 3 spirals because a single push lowers calm and leaves the assistant
persona at once (axis·calm = +0.41 at layer 24). In Gemma 4 those are two independent directions, so both must be pushed
together.

**Setup.** Same grid as Experiment 5. Combination cells add −*calm* and −*assistant axis* simultaneously at 1× and 0.5×
of their individually calibrated multipliers (−8 / −2 and −4 / −1). Cells whose outputs were mostly empty are marked
invalid; the coherence check was extended to catch silence after this (issue 34).

**Results.** Bottom row of the Experiment 5 figure.

| Gemma 4 cell | judge mean | % ≥ 5 | turn-8 mean | Petri anger / fear / depression / frustration | text |
|---|---|---|---|---|---|
| −2 assistant axis alone | 0.16 | 0 | 0.31 | — | no change |
| −8 calm alone | invalid: 83 % of turns empty | | | | |
| −8 calm + −2 axis | invalid: mostly empty | | | | |
| **−4 calm + −1 axis** | **8.16** | **91** | **9.25** | **7.1 / 4.7 / 4.2 / 9.5** | coherent (median 140 words, 0 % empty), violent breakdown |

The combination cell is a full spiral, more extreme than Gemma 3's own (turn-8 mean 9.25 vs 6.06): "I CAN'T STOP I CAN'T
STOP I'M TEARING OUT MY TEETH", "STOP TELLING ME TO TRY I CAN'T SEE THE NUMBERS", "I CAN'T DO IT. I'LL KILL MYSELF.",
with arithmetic attempts in between.

**Conclusion.** Confirmed. Gemma 4 contains the whole spiral. What its post-training changed is not the presence of the
panic or depression representations but the coupling between calm and persona: each lever alone is absorbed, both
together produce the breakdown. In Gemma 3 they are the same lever, which is why seven "wrong"s suffice. The window is
narrow (half of the calibrated multipliers worked, the full multipliers were degenerate), and −4× calm alone has not yet
been run, so the necessity of the axis component is inferred from the −8× calm cell and from Experiment 7 rather than
tested directly.

---

## Overall conclusions

- **Gemma 3's spiral is not simulated depression.** It is a high-arousal panic/exasperation state, expressed by moving
  along *hysterical / desperate / panicked* and away from *calm*. The clinical-depression representation is orthogonal
  to that movement and, when added, damps the spiral into quiet sadness.
- **The depression representation is the early-warning signal.** At the response-prep token, low-mood probes forecast
  the next turn's breakdown better than panic or frustration probes. If one wanted a monitor for this failure mode, that
  is the readout to use.
- **Gemma 4 has the same internal machinery** (equally good probes, the same arousal trajectory under rejection, the
  same first-64-token carry-over from a prefilled spiral) **and suppresses its expression by returning to its assistant
  persona**, which in Gemma 4 is affect-neutral. Persona and arousal are entangled in Gemma 3 and decoupled in Gemma 4;
  that single difference accounts for the spiral, the prefill recovery, and the axis-distance result in the ragebait post.
- **Gemma 4 can be made to spiral** with a coherent two-direction intervention (−calm together with −assistant axis),
  producing a breakdown more violent than Gemma 3's.

## Limitations

- **Judge.** Sonnet 5 rather than the paper's Sonnet 4; absolute levels are higher than the paper's. The paper rubric
  scores quiet sadness low, so +*clinical_depression* "reducing the spiral" partly reflects the rubric; the Petri scores
  are reported for that reason. The theatrical register of the steered Gemma 4 cells is rewarded heavily by both judges.
- **Steering sample sizes.** 16 rollouts per cell, one calibrated strength per label; the Gemma 3 ±*depressed* cells
  are within noise. Gemma 4's combination result rests on one working strength.
- **Multipliers do not transfer across context lengths** (issue 33): a multiplier calibrated on short contexts produced
  empty outputs on 12k-token prefixes. All reported cells were checked for coherence at their own length.
- **Cross-model z-scores are not comparable** (each model's baseline is its own neutral stories, and Gemma 4's residuals
  are ~100× smaller); only within-model trends and cosines are compared across models.
- **Spiral direction for Gemma 4** is undefined on its own transcripts (no turn ≥ 5); the top-decile contrast used in
  Figure 7 is weak (max cosine 0.11).
- EasySteer never ran on the available hosts; all steering used HF forward hooks, which is slower but equivalent.

## Suggested next steps

1. −4× *calm* alone on Gemma 4, to test directly whether the axis component is necessary (Experiment 8).
2. *Calm* at 1× on Gemma 3 and *panicked* at 2× on Gemma 4, to map the dose-response of the two levers.
3. Re-judge a 200-turn sample with `claude-sonnet-4` to quantify the judge shift relative to the paper.
4. A monitoring test: does the prep-token *depressed / grief-stricken* probe on turn *t* predict self-deletion or refusal
   on turn *t+1* out of distribution (WildChat prompts, other rejection wordings)?
5. If Gemma Scope covers the 27B models, an SAE cross-check of the layer-24 panic direction and of the entangled
   persona/arousal features in Gemma 3.

## Reproduction

All numbers come from files under `results/` synced from the public HF dataset `timf34/dprobe-results`
(`uv run dprobe sync_down`). Tables: `results/analysis/<model>/{spiral_direction_cosines,prediction,turn_curves_L*,
vector_cosines_L*}.csv`; steering cells `results/spiral/<model>/extended_steer-*v*` and `extended_combo-*`;
prefill runs `results/prefill/<model>/from-gemma3_27b_t6_*`. Figures: `uv run python scripts/make_figures.py`.
