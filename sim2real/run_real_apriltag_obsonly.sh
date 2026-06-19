#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
if [[ ! "$VERSION" =~ ^v(21|22|23|24)$ ]]; then
  echo "usage: bash sim2real/run_real_apriltag_obsonly.sh v21|v22|v23|v24" >&2
  exit 2
fi

CONFIG="sim2real/configs/g1_catch_real_urop_${VERSION}_apriltag_obsonly.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 1
fi

for path in \
  sim2real/calib/head_camera_extrinsics.real.yaml \
  sim2real/calib/head_camera_intrinsics.real.yaml \
  sim2real/calib/box_tag_config.real.yaml
do
  if [[ ! -f "$path" ]]; then
    echo "missing required calibration file: $path" >&2
    echo "obsonly is blocked until real AprilTag calibration files are present." >&2
    exit 1
  fi
done

PRINT_RATE="${PRINT_RATE:-10}"
START_POSE="${START_POSE:-catch_ready}"

check_cmd=(python sim2real/g1_catch_real.py --check-only --config "$CONFIG" --no-policy)
echo "[sim2real] check-only: ${check_cmd[*]}"
"${check_cmd[@]}"

cmd=(python sim2real/g1_catch_real.py --config "$CONFIG" --no-policy --start-pose "$START_POSE" --print-rate "$PRINT_RATE")
if [[ -n "${NET_IFACE:-}" ]]; then
  cmd+=(--net-iface "$NET_IFACE")
fi
if [[ -n "${MOVE_DURATION:-}" ]]; then
  cmd+=(--move-duration "$MOVE_DURATION")
fi

echo "[sim2real] running obs-only command:"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
