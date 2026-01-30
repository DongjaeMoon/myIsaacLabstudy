#!/usr/bin/env bash
set -e

# -----------------------
# default args
# -----------------------
NUM_ENVS=64
HEADLESS="--headless"
ITERS0=2000
ITERS1=4000
ITERS2=8000

# -----------------------
# arg parsing
# -----------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_envs) NUM_ENVS="$2"; shift 2;;
    --iters0) ITERS0="$2"; shift 2;;
    --iters1) ITERS1="$2"; shift 2;;
    --iters2) ITERS2="$2"; shift 2;;
    --headless) HEADLESS="--headless"; shift;;
    --gui) HEADLESS=""; shift;;
    *) shift;;
  esac
done


ISAACLAB_ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAACLAB_SH="${ISAACLAB_ROOT}/isaaclab.sh"
TRAIN_SCRIPT="${ISAACLAB_ROOT}/UROP/train_rsl_rl.py"

run_stage () {
  STAGE=$1
  ITERS=$2
  RUN_NAME="S${STAGE}_teacher"

  echo "=============================="
  echo "[RUN] stage=${STAGE} iters=${ITERS} num_envs=${NUM_ENVS}"
  echo "=============================="

  export UROP_STAGE=${STAGE}
  export UROP_MODE=teacher

  ${ISAACLAB_SH} -p ${TRAIN_SCRIPT} \
    --task Isaac-Urop-v0 \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${ITERS} \
    --run_name ${RUN_NAME} \
    ${HEADLESS}
}

run_stage 0 ${ITERS0}
run_stage 1 ${ITERS1}
run_stage 2 ${ITERS2}

echo "🎉 All curriculum stages finished"

