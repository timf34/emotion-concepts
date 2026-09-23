#!/usr/bin/env bash
# Stop (or terminate) THIS pod via the RunPod REST API so an unattended run never keeps billing.
#   SHUTDOWN=stop|terminate bash pod/self_stop.sh     (no-op when SHUTDOWN is unset)
# Needs RUNPOD_API_KEY (from .env) and RUNPOD_POD_ID (set automatically inside RunPod pods).
set -uo pipefail
[ -n "${SHUTDOWN:-}" ] || exit 0
# ssh/tmux sessions do not inherit the container env; read the pod id from PID 1's environment if needed.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ]; then
  RUNPOD_POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)
fi
if [ -z "${RUNPOD_POD_ID:-}" ]; then RUNPOD_POD_ID=$(hostname | grep -oE '^[a-z0-9]{14}$' || true); fi
[ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || { echo "!! self-stop skipped: RUNPOD_API_KEY or RUNPOD_POD_ID missing"; exit 0; }
if [ "$SHUTDOWN" = "terminate" ]; then
  echo "== SHUTDOWN=terminate -> DELETE pod $RUNPOD_POD_ID =="
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"
else
  echo "== SHUTDOWN=stop -> stop pod $RUNPOD_POD_ID =="
  curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
fi
echo
