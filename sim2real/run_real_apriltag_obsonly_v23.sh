#!/usr/bin/env bash
set -euo pipefail

echo "[sim2real] DEPRECATED wrapper: use bash sim2real/run_real_apriltag_obsonly.sh v23" >&2
exec bash sim2real/run_real_apriltag_obsonly.sh v23
