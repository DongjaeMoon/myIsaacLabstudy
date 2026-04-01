#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/idim5080-2/mdj/myIsaacLabstudy"
POLICY="$ROOT/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt"
NET_IFACE="${1:-YOUR_INTERFACE}"

cd "$ROOT"
python3 sim2real/g1_catch_real.py \
  --policy "$POLICY" \
  --net-iface "$NET_IFACE"
