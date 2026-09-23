#!/usr/bin/env bash
# Phase 0 for one model: stories -> spiral transcripts -> judge (frustration + petri). Resumable; safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
M=$1
LOG=results/logs/phase0_$M.log
{
  echo "=== phase0 $M start $(date) ==="
  uv run python -m dprobe.cli stories "$M" --sets neutral,syndromes,emotions
  uv run python -m dprobe.cli spiral "$M" --rollouts 200 --extra 10
  uv run python -m dprobe.cli judge "$M" --rubric frustration
  uv run python -m dprobe.cli judge "$M" --rubric petri
  uv run python -m dprobe.cli summary "$M"
  echo "=== phase0 $M end $(date) ==="
} >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
