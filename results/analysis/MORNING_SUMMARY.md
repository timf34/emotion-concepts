# Overnight results — 2026-09-24

Full details, tables and figures: `results/analysis/NOTES.md` (chronological), `results/analysis/<model>/`.
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
   (max cos 0.13). Steering Gemma 4 toward depression (positive multipliers only) is the last run in
   progress; the first (over-driven) attempt lifted its judge mean from 0.08 to 1.4 with borderline text.
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
- Steer with the *hysterical* / *panicked* vectors (the spiral's own direction) and with *calm* at 1x.
- Re-judge a 200-turn sample with claude-sonnet-4 to quantify the judge shift vs the paper.
- SAE cross-check of the panic direction if Gemma Scope covers 27B.
