#!/usr/bin/env bash
set -euo pipefail

NET_IFACE="${NET_IFACE:-enx00e04c0f3e58}"
PRINT_RATE="${PRINT_RATE:-10}"

sudo ip link set "${NET_IFACE}" multicast on

python sim2real/g1_catch_real.py \
  --config sim2real/configs/g1_catch_real_real_apriltag_obsonly_traingain_v23.yaml \
  --net-iface "${NET_IFACE}" \
  --start-pose catch_ready \
  --no-policy \
  --print-rate "${PRINT_RATE}"
