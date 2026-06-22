#!/usr/bin/env bash
set -euo pipefail

# WARNING:
#   Run this only in the lab, with the robot area clear and an operator ready.
#   This command sends LowCmd to the real robot.

ROOT="/home/dongjae/myIsaacLabstudy/sim2real"
PYTHON="${PYTHON:-/home/dongjae/miniconda3/envs/urop/bin/python}"
NET_IFACE="${1:-enx00e04c0f3e58}"

cd "$ROOT"
"$PYTHON" g1_catch_real.py \
  --config configs/g1_catch_real_real_apriltag_obsonly_traingain.yaml \
  --net-iface "$NET_IFACE" \
  --start-pose catch_ready \
  --no-policy \
  --print-rate 10
