# gpt-oss-120b: emotion vectors + re-encoded assistant axis

Run 2026-10-20 on 2x H200 (`pod/launch_rest.py gptoss_120b`, ~50 min including a smoke pass). Same recipe as
gpt-oss-20b: Gemma 3's stories, MXFP4 dequantized to bf16, eager attention, re-encoded axis from Gemma 3's
archived role-play (275 roles x 50 + 300 default, "You are ChatGPT."). Outputs: HF `timf34/dprobe-results`,
`vectors/gptoss_120b/` (42 emotions, 6 syndromes, axis as external vector `assistant_axis`).

Numbers below come from `scripts/analyze_crossfamily.py` run without the published Gemma axes (they live on the
laptop), so the committed `entanglement.csv` / `robust.csv` / figure are not regenerated yet:
`GEMMA_AXIS_DIR=<dir> uv run python scripts/analyze_crossfamily.py` refreshes them with the 120b row included.

**Vectors work.** Held-out one-vs-rest AUC, best layer, denoised: median 0.929 (min 0.827, `dispirited`) vs 0.923
for gpt-oss-20b; most labels peak at layers 12-14 of 36. Syndromes 0.93-1.00.

| | gpt-oss-20b | gpt-oss-120b | Qwen3 32B | GLM-4.5-Air |
|---|---|---|---|---|
| median \|cos\| of axis with emotions, 25-42% depth | 0.09 | 0.05 | 0.04 | 0.03 |
| ... as a multiple of chance | 6.3x | 3.4x | 3.4x | 2.3x |
| calm, toward the assistant end | +0.15 | +0.07 | +0.04 | -0.02 |
| hopeful, toward the assistant end | +0.16 | +0.10 | +0.05 | +0.03 |
| personas' mean hysteria relative to the assistant | +0.02 | +0.01 | -0.03 | 0.00 |
| toddler: hysterical | +0.15 | +0.16 | +0.14 | +0.13 |
| \|cos\| with every coordinate standardised | 0.038 | 0.039 | 0.025 | 0.027 |
| calm-end AUC (null: random persona pairs) | 0.67 (0.63) | 0.67 (0.65) | 0.73 (0.69) | 0.75 (0.65) |

Reading: gpt-oss-120b has the same faint calm/hopeful lean at its assistant end as 20b, about half as strong by
cosine, and the same once coordinates are standardised (0.039 vs 0.038). Like every non-Gemma model it is far from
Gemma 3 (0.28, calm +0.37, personas +0.24). No behaviour (rejection-eval) run for 120b yet.
