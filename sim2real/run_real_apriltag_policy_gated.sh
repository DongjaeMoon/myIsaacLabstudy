#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
POLICY_PATH="${2:-}"
if [[ ! "$VERSION" =~ ^v(21|22|23|24)$ ]] || [[ -z "$POLICY_PATH" ]]; then
  echo "usage: bash sim2real/run_real_apriltag_policy_gated.sh v21|v22|v23|v24 logs/rsl_rl/UROP_vXX/<run>/exported/policy.pt" >&2
  exit 2
fi

CONFIG="sim2real/configs/g1_catch_real_urop_${VERSION}_apriltag_policy_gated.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$POLICY_PATH" ]]; then
  echo "missing policy: $POLICY_PATH" >&2
  exit 1
fi
if [[ "$(basename "$POLICY_PATH")" == model_*.pt ]]; then
  echo "refusing raw checkpoint $POLICY_PATH; use exported/policy.pt TorchScript" >&2
  exit 1
fi
if [[ "$POLICY_PATH" != */exported/policy.pt ]]; then
  echo "WARNING: policy path is not */exported/policy.pt: $POLICY_PATH" >&2
fi

for path in \
  sim2real/calib/head_camera_extrinsics.real.yaml \
  sim2real/calib/head_camera_intrinsics.real.yaml \
  sim2real/calib/box_tag_config.real.yaml
do
  if [[ ! -f "$path" ]]; then
    echo "missing required calibration file: $path" >&2
    echo "gated policy is blocked until real AprilTag calibration files are present." >&2
    exit 1
  fi
done

PRINT_RATE="${PRINT_RATE:-10}"
START_POSE="${START_POSE:-catch_ready}"

check_cmd=(python sim2real/g1_catch_real.py --check-only --config "$CONFIG" --use-policy --policy "$POLICY_PATH")
echo "[sim2real] check-only: ${check_cmd[*]}"
"${check_cmd[@]}"

cmd=(python sim2real/g1_catch_real.py --config "$CONFIG" --use-policy --policy "$POLICY_PATH" --start-pose "$START_POSE" --print-rate "$PRINT_RATE")
if [[ -n "${NET_IFACE:-}" ]]; then
  cmd+=(--net-iface "$NET_IFACE")
fi
if [[ -n "${MOVE_DURATION:-}" ]]; then
  cmd+=(--move-duration "$MOVE_DURATION")
fi

echo "[sim2real] running gated policy command:"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
