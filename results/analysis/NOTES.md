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
