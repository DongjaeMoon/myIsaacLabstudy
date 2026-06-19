#!/usr/bin/env bash
set -euo pipefail

NET_IFACE="${NET_IFACE:-enx00e04c0f3e58}"
PRINT_RATE="${PRINT_RATE:-10}"
POLICY_PATH="${1:-}"

if [[ -z "${POLICY_PATH}" ]]; then
  echo "Usage: $0 <path/to/UROP_v23/exported/policy.pt>" >&2
  echo "Example: NET_IFACE=enx00e04c0f3e58 $0 logs/rsl_rl/UROP_v23/<run>/exported/policy.pt" >&2
  exit 2
fi

sudo ip link set "${NET_IFACE}" multicast on

python sim2real/g1_catch_real.py \
  --config sim2real/configs/g1_catch_real_real_apriltag_policy_gated_traingain_v23.yaml \
  --net-iface "${NET_IFACE}" \
  --start-pose catch_ready \
  --use-policy \
  --policy "${POLICY_PATH}" \
  --print-rate "${PRINT_RATE}"
