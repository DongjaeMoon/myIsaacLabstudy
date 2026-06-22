#!/usr/bin/env bash
set -euo pipefail

# Real robot command. Run only if the AprilTag image/ZMQ pipeline is already stable.
# Policy remains gated until a visible object observation is received.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
NET_IFACE="${1:-enx00e04c0f3e58}"

cd "$ROOT"
"$PYTHON" g1_catch_real.py \
  --config configs/g1_catch_demo_apriltag_gated_soft.yaml \
  --net-iface "$NET_IFACE" \
  --start-pose catch_ready \
  --move-duration 4.0 \
  --use-policy \
  --print-rate 10
