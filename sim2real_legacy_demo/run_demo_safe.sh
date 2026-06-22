#!/usr/bin/env bash
set -euo pipefail

# Real robot command. Run only with the G1 area clear and an operator ready.
# Policy is disabled; this is the safest scripted/interactive upper-body demo route.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
NET_IFACE="${1:-enx00e04c0f3e58}"

cd "$ROOT"
"$PYTHON" g1_catch_config_demo.py \
  --config configs/g1_catch_demo_stable_soft.yaml \
  --net-iface "$NET_IFACE" \
  --start-pose safe_stand \
  --move-duration 4.0 \
  --interactive \
  --print-rate 5
