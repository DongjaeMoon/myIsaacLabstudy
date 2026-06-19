#!/usr/bin/env bash
set -euo pipefail

# Safe standalone AprilTag pipeline test.
# This script does not use DDS and does not send robot commands.

ROOT="/home/dongjae/myIsaacLabstudy/sim2real"
PYTHON="${PYTHON:-/home/dongjae/miniconda3/envs/urop/bin/python}"

cd "$ROOT"
"$PYTHON" test_apriltag_zmq.py \
  --server-address 192.168.123.164 \
  --port 5555 \
  --intrinsics-yaml calib/head_camera_intrinsics.real.yaml \
  --extrinsics-yaml calib/head_camera_extrinsics.real.yaml \
  --tag-yaml calib/box_tag_config.real.yaml \
  --print-rate 5 \
  "$@"
