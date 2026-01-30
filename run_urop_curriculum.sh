#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# paths (script location-safe)
# -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_SH="${SCRIPT_DIR}/isaaclab.sh"
TRAIN_PY="${SCRIPT_DIR}/UROP/train_rsl_rl.py"
LOG_ROOT="${SCRIPT_DIR}/logs/rsl_rl/UROP_v0"

# -----------------------
# defaults (you can override by args)
# -----------------------
NUM_ENVS=64
ITER0=2000
ITER1=4000
ITER2=8000
HEADLESS=1        # 1=headless, 0=GUI
VIDEO=0           # 1=record video

# -----------------------
# arg parsing
# -----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --iters0)   ITER0="$2"; shift 2 ;;
    --iters1)   ITER1="$2"; shift 2 ;;
    --iters2)   ITER2="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --gui)      HEADLESS=0; shift ;;
    --video)    VIDEO=1; shift ;;
    --no-video) VIDEO=0; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--num_envs N] [--iters0 I] [--iters1 I] [--iters2 I] [--headless|--gui] [--video|--no-video]"
      exit 1
      ;;
  esac
done

extra_flags=()
if [[ "${HEADLESS}" -eq 1 ]]; then
  extra_flags+=(--headless)
fi
if [[ "${VIDEO}" -eq 1 ]]; then
  # tweak to taste
  extra_flags+=(--video --video_length 150 --video_interval 200)
fi

# -----------------------
# helpers
# -----------------------
latest_run_name() {
  # newest run folder name (basename)
  ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1 | xargs -n1 basename
}

latest_ckpt_name() {
  local run="$1"
  # newest checkpoint filename inside that run
  ls -1t "${LOG_ROOT}/${run}/checkpoints"/*.pt 2>/dev/null | head -n 1 | xargs -n1 basename
}

run_stage() {
  local stage="$1"
  local iters="$2"
  local run_name="$3"
  local resume="$4"      # "yes" or "no"
  local load_run="$5"    # folder name
  local ckpt="$6"        # checkpoint filename

  export UROP_STAGE="${stage}"
  export UROP_MODE="teacher"

  cmd=( "${ISAACLAB_SH}" -p "${TRAIN_PY}"
        --task Isaac-Urop-v0
        --num_envs "${NUM_ENVS}"
        --max_iterations "${iters}"
        --run_name "${run_name}"
      )

  cmd+=( "${extra_flags[@]}" )

  if [[ "${resume}" == "yes" ]]; then
    # NOTE: your cli uses --resume/--load_run/--checkpoint (not --load_checkpoint)
    cmd+=( --resume True --load_run "${load_run}" --checkpoint "${ckpt}" )
  fi

  echo ""
  echo "=============================="
  echo "[RUN] stage=${stage} iters=${iters} num_envs=${NUM_ENVS} run_name=${run_name}"
  if [[ "${resume}" == "yes" ]]; then
    echo "[RESUME] from run=${load_run}, ckpt=${ckpt}"
  fi
  echo "CMD: ${cmd[*]}"
  echo "=============================="
  echo ""

  "${cmd[@]}"
}

# -----------------------
# main: Stage0 -> Stage1 -> Stage2
# -----------------------
mkdir -p "${LOG_ROOT}"

# Stage0: balance/recovery (fresh)
run_stage 0 "${ITER0}" "S0_teacher" "no" "-" "-"

# Stage1: gentle catch (resume from Stage0's latest checkpoint)
RUN0="$(latest_run_name)"
CKPT0="$(latest_ckpt_name "${RUN0}")"
if [[ -z "${RUN0}" || -z "${CKPT0}" ]]; then
  echo "[ERR] Could not find Stage0 run/checkpoint under ${LOG_ROOT}"
  exit 2
fi
run_stage 1 "${ITER1}" "S1_teacher" "yes" "${RUN0}" "${CKPT0}"

# Stage2: full catch + shock mitigation (resume from Stage1's latest checkpoint)
RUN1="$(latest_run_name)"
CKPT1="$(latest_ckpt_name "${RUN1}")"
if [[ -z "${RUN1}" || -z "${CKPT1}" ]]; then
  echo "[ERR] Could not find Stage1 run/checkpoint under ${LOG_ROOT}"
  exit 3
fi
run_stage 2 "${ITER2}" "S2_teacher" "yes" "${RUN1}" "${CKPT1}"

echo ""
echo "[DONE] Curriculum finished. Latest run: $(latest_run_name)"

