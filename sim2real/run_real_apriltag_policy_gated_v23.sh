#!/usr/bin/env bash
set -euo pipefail

echo "[sim2real] DEPRECATED wrapper: use bash sim2real/run_real_apriltag_policy_gated.sh v23 <exported/policy.pt>" >&2
exec bash sim2real/run_real_apriltag_policy_gated.sh v23 "$@"
