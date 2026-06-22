#!/usr/bin/env bash
set -euo pipefail

# Real robot command. Run only with the G1 area clear and an operator ready.
# Loads the conservative v15 policy, uses a built-in fake object, and freezes lower body/waist targets to current q.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
NET_IFACE="${1:-enx00e04c0f3e58}"

cd "$ROOT"
"$PYTHON" g1_catch_real.py \
  --config configs/g1_catch_demo_fake_object_soft.yaml \
  --net-iface "$NET_IFACE" \
  --start-pose catch_ready \
  --move-duration 4.0 \
  --use-policy \
  --print-rate 10
