#!/usr/bin/env bash
# Stop (or terminate) THIS pod via the RunPod REST API so an unattended run never keeps billing.
#   SHUTDOWN=stop|terminate bash pod/self_stop.sh     (no-op when SHUTDOWN is unset)
# Needs RUNPOD_API_KEY (from .env) and RUNPOD_POD_ID (set automatically inside RunPod pods).
set -uo pipefail
[ -n "${SHUTDOWN:-}" ] || exit 0
[ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || { echo "!! self-stop skipped: RUNPOD_API_KEY or RUNPOD_POD_ID missing"; exit 0; }
if [ "$SHUTDOWN" = "terminate" ]; then
  echo "== SHUTDOWN=terminate -> DELETE pod $RUNPOD_POD_ID =="
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"
else
  echo "== SHUTDOWN=stop -> stop pod $RUNPOD_POD_ID =="
  curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
fi
echo
