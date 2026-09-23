#!/usr/bin/env bash
# Create a pod, trying GPU types in order until one is available.
#   bash pod/up.sh dprobe-smoke                    # h200 -> h100 -> a100, no network volume
#   GPUS="h200 h100" VOLUME=<id> bash pod/up.sh dprobe-x
set -uo pipefail
NAME=$1
GPUS=${GPUS:-"h200 h100 a100"}
VOLUME=${VOLUME:-none}
DISK=${DISK:-140}
RP=${RP:-rp}
for g in $GPUS; do
  echo "[up] trying $g for $NAME ..."
  if $RP up --name "$NAME" --gpu "$g" --volume "$VOLUME" --disk "$DISK"; then
    echo "[up] $NAME created on $g"; exit 0
  fi
  sleep 5
done
echo "[up] no GPU available for $NAME (tried: $GPUS)"; exit 1
