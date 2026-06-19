#!/usr/bin/env bash
set -euo pipefail

echo "[sim2real] DEPRECATED wrapper: use ALLOW_NOGATE=1 bash sim2real/run_real_apriltag_policy_nogate.sh v23 <exported/policy.pt>" >&2
exec bash sim2real/run_real_apriltag_policy_nogate.sh v23 "$@"
