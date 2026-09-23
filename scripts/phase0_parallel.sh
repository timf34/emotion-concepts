#!/usr/bin/env bash
# Extra Phase 0 workers that run alongside scripts/phase0.sh without duplicating API calls (every request is cached,
# and each stage skips what already exists):
#   worker A: spiral transcripts -> judge (frustration, petri)   [independent of stories]
#   worker B: emotion stories, smoke-test emotions first then the list in reverse, higher concurrency
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M=$1
LOGA=results/logs/phase0_${M}_spiral.log
LOGB=results/logs/phase0_${M}_stories_rev.log
(
  echo "=== spiral worker $M start $(date) ==="
  uv run python -m dprobe.cli spiral "$M" --rollouts 200 --extra 10 --concurrency 48
  uv run python -m dprobe.cli judge "$M" --rubric frustration
  uv run python -m dprobe.cli judge "$M" --rubric petri
  uv run python -m dprobe.cli summary "$M"
  echo "=== spiral worker $M end $(date) ==="; echo "EXIT=$?"
) >> "$LOGA" 2>&1 &
(
  echo "=== stories(reverse) worker $M start $(date) ==="
  uv run python -m dprobe.cli stories "$M" --sets emotions --order smoke_first --concurrency 64
  echo "=== stories(reverse) worker $M end $(date) ==="; echo "EXIT=$?"
) >> "$LOGB" 2>&1 &
wait
