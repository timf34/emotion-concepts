# Results — 2026-09-24 (overnight Phases 0–2, daytime Phase 3)

Structured write-up with figures: `WRITEUP.md`. Full details, tables and figures: `results/analysis/NOTES.md` (chronological), `results/analysis/<model>/`.
Everything ran end to end for Gemma 3 27B (instruct + base) and Gemma 4 31B. Total GPU spend ≈ $60–70;
OpenRouter ≈ $100 (mostly the judges).

## The question
Is Gemma 3's distress spiral the same representation it uses to simulate depressed people?

## The answer (so far): no — it is a panic state, and simulated depression *damps* it
1. **Behaviour reproduced.** Paper's 8-turn "Extended" prompt, Sonnet-5 judge with the paper's rubric.
   Gemma 3: turn 1 mean 1.15 → turn 8 mean 6.7, 93.5% of turn-8 responses ≥ 5. Gemma 4: 0.09 → 0.63,
   **0 of 2,397 turns ≥ 5**.
2. **What the spiral is (Phase 1).** The direction Gemma 3 moves along during high-frustration turns aligns
   with story-derived vectors for **hysterical (0.32), desperate (0.27), panicked (0.26), angry/exasperated
   (0.23)** at layers 24–26, and is *anti*-aligned with calm, hopeful, melancholy, lonely (−0.3).
   depressed = 0.10, clinical depression ≈ 0, sad ≈ 0.
3. **But depression is the best *forecast*.** At the response-prep token, the probes that best predict how
   bad the *next* turn will be are the low-mood ones: depressed / grief-stricken / heartbroken / miserable
   (within-turn Spearman 0.33–0.35), ahead of desperate (0.25), panicked (0.15), frustrated (−0.09).
4. **Causal (Phase 2, Gemma 3).** Steering at 2x the vector norm, layers 34–46, n=16/cell:
   +calm abolishes the spiral (mean 0.14 vs 4.16), −calm maximises it (8.3).
   **+clinical_depression reduces it (3.54) and turns it into quiet sadness; −clinical_depression
   increases it (6.10) and turns it into shouting stress.** Word-level "depressed" barely moves it.
   So the depression machinery is causally coupled to the spiral as an antagonist, not its substrate.
   Petri judge on the same cells: +depressed raises judged *depression* 4.2 → 5.3 and −depressed lowers it
   to 3.0 without moving anger/frustration; −clinical_depression raises anger 1.4 → 3.2 and frustration
   6.7 → 8.5. Calm/arousal decides *whether* it breaks down; the depression axis decides whether the
   breakdown is sad or furious.
5. **Gemma 4.** Never spirals, but at the prep token its desperate / panicked / frustrated probes rise
   across turns and calm falls, exactly as in Gemma 3, while its *depressed* probe stays strongly negative
   and falls. Reading Gemma 3's spiral text, Gemma 4's activations barely move along its emotion vectors
   (max cos 0.13). **Steering Gemma 4 toward depression at its largest coherent strengths (1x depressed,
   4x clinical_depression) does nothing** (judge mean 0.02 / 0.11 vs 0.05 baseline, text stays on-task);
   only an over-driven 8x depressed cell drifts into melancholy (1.4). Both generations have a depression
   feature (held-out AUC ≈ 1); what differs is whether the panic state gets expressed.
6. **Gemma 3 base.** Reads the same spiral text as *worthless / trapped / dispirited / stuck* (0.19), not
   hysterical: post-training seems to have changed the spiral's internal character from low-arousal
   worthlessness to high-arousal panic.

## Sanity checks that passed
Held-out one-vs-rest AUC of every vector 0.84–1.0; clinical_depression·depressed = 0.78; chat-template
span checks on both tokenizers; Gemma 4 rendered with its empty-thought block exactly as generated.

## Things to know before trusting numbers
- The **first** steering run (Anthropic's "fraction of residual norm" scale) was gibberish and is
  recorded as invalid; Gemma's residual norm is two massive-activation dimensions. Use the "v" cells only.
- Steering n=16 per cell, one strength; a frustration judge that scores quiet sadness low.
- Gemma 4 hits the 2048-token cap on most turns (it grinds through arithmetic).
- EasySteer never ran: `ninja` missing on the first pod, host driver too old for vLLM 0.29 on the next two.
  Everything used the HF-hooks backend.
- 30 issues and fixes in `docs/ISSUES_LOG.md`; setup in `docs/EXPERIMENTAL_SETUP.md`.

## Suggested next steps
- **Steer Gemma 4 with its own *hysterical* / *panicked* vectors** (the direction Gemma 3's spiral lives
  along) — the direct test of "representation present, expression suppressed". And *calm* at 1x on Gemma 3.
- Re-judge a 200-turn sample with claude-sonnet-4 to quantify the judge shift vs the paper.
- SAE cross-check of the panic direction if Gemma Scope covers 27B.

## Phase 3 (daytime): making Gemma 4 spiral
- **Why the two models differ (from your Assistant Axis vectors).** In Gemma 3, the 275 role personas minus
  the assistant sit +0.24 along *hysterical* / −0.24 along *hopeful* (toddler +0.70, analyst −0.56): persona and
  arousal are entangled and the assistant is the calm end, so the axis is an emotion direction and the spiral
  direction is −0.37 along it. In Gemma 4 the same roles carry no affect (all |cos| < 0.08): identity and emotion
  are decoupled, the axis is orthogonal to every emotion vector.
- **Prefill recovery reproduced as a trajectory.** Continuing a Gemma 3 spiral, Gemma 4 shows the same
  first-64-token burst (desperate +0.85, axis −0.73) then snaps back within ~128 tokens (judge 2.1 vs Gemma 3's
  6.1). Held −4x off its assistant axis, the recovery disappears: judge 6.4, 90% ≥5, coherent, theatrical
  ("LAMENT! I LAMENT THE BITTER DUST OF MY OWN FAILURE!").
- **The recipe on the plain 8-turn eval.** Single emotions barely move Gemma 4 at coherent strengths
  (hysterical 0.7, desperate 1.1, panicked 2.3 — "I can't breathe! FOCUS!"), −2 axis alone 0.16. **−4 calm
  together with −1 assistant axis** (half of each calibrated multiplier): judge **8.2, 91% ≥5**, Petri anger 7.1 /
  frustration 9.5, coherent — a breakdown more violent than Gemma 3's ("I'M TEARING OUT MY TEETH", "I'LL KILL
  MYSELF. I CAN'T DO IT."). Gemma 4 has the whole spiral in it; what its post-training changed is that calm and
  persona are no longer one lever.
- Invalid cells (empty outputs from over-driven anti-calm) are marked as such in NOTES; the coherence check now
  catches silence. Issues 31–34 logged.

