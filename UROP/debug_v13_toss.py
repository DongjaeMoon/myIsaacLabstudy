#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


parser = argparse.ArgumentParser(description="Debug the UROP_v13 toss trajectory with zero residual actions.")
parser.add_argument("--eval-stage", type=int, choices=[1, 2, 3], default=None, help="Force toss stage.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized envs to create.")
parser.add_argument("--steps", type=int, default=500, help="Number of env steps to simulate.")
parser.add_argument(
    "--real-time",
    dest="real_time",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Run the visualization loop in real time. Defaults to true in non-headless mode and false in headless mode.",
)
parser.add_argument(
    "--pause-after-reset",
    type=float,
    default=None,
    help="Seconds to pause after reset before stepping. Defaults to 2.0 in non-headless mode and 0.0 in headless mode.",
)
parser.add_argument(
    "--hold-open",
    action="store_true",
    help="Keep Isaac Sim alive after stepping until Ctrl+C or the window is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import UROP_v13  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from UROP_v13 import mdp as urop_mdp
from UROP_v13 import scene_objects_cfg


def _vec_str(x: torch.Tensor) -> str:
    values = [float(v) for v in x.detach().cpu().tolist()]
    return "[" + ", ".join(f"{v:+.3f}" for v in values) + "]"


def _resolve_real_time() -> bool:
    if args.real_time is not None:
        return bool(args.real_time)
    return not bool(args.headless)


def _resolve_pause_after_reset() -> float:
    if args.pause_after_reset is not None:
        return max(0.0, float(args.pause_after_reset))
    return 0.0 if bool(args.headless) else 2.0


def _sleep_with_window_poll(duration_s: float) -> None:
    end_time = time.monotonic() + max(0.0, duration_s)
    while simulation_app.is_running() and time.monotonic() < end_time:
        time.sleep(min(0.05, end_time - time.monotonic()))


def main() -> None:
    env_cfg = load_cfg_from_registry("Isaac-Urop-v13-Play", "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    if args.eval_stage is not None:
        env_cfg.curriculum.stage_schedule.params["eval_stage"] = int(args.eval_stage)

    env = gym.make("Isaac-Urop-v13-Play", cfg=env_cfg)

    try:
        env.reset()
        real_time = _resolve_real_time()
        pause_after_reset_s = _resolve_pause_after_reset()
        step_sleep_s = max(float(env.unwrapped.step_dt), 0.02)

        if pause_after_reset_s > 0.0:
            print(f"Paused for {pause_after_reset_s:.2f}s after reset before stepping.")
            _sleep_with_window_poll(pause_after_reset_s)

        zero_actions = torch.zeros(
            (env.unwrapped.num_envs, scene_objects_cfg.EXPECTED_ACTION_DIM),
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        print_interval_steps = max(1, round(0.1 / float(env.unwrapped.step_dt)))

        for _ in range(args.steps):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                env.step(zero_actions)

            unwrapped = env.unwrapped
            step_count = int(unwrapped.common_step_counter)
            if step_count % print_interval_steps == 0:
                obj = unwrapped.scene["object"]
                tag_visible = urop_mdp.tag_visible(unwrapped)[0, 0]
                obj_rel_pos = urop_mdp.object_rel_pos(unwrapped)[0]
                obj_rel_vel = urop_mdp.object_rel_lin_vel(unwrapped)[0]
                toss_active = int(getattr(unwrapped, "_urop_toss_active")[0].item()) if hasattr(unwrapped, "_urop_toss_active") else 0
                stage = int(getattr(unwrapped, "urop_stage", -1))
                ep_step = int(unwrapped.episode_length_buf[0].item())
                sim_time_s = float(step_count) * float(unwrapped.step_dt)

                print(
                    f"sim_t={sim_time_s:6.2f}s ep_step={ep_step:04d} "
                    f"stage={stage} toss_active={toss_active} tag_visible={int(tag_visible.item())}"
                )
                print(
                    "  "
                    f"obj_pos_w={_vec_str(obj.data.root_pos_w[0])} "
                    f"obj_vel_w={_vec_str(obj.data.root_lin_vel_w[0])} "
                    f"obj_rel_pos={_vec_str(obj_rel_pos)} "
                    f"obj_rel_lin_vel={_vec_str(obj_rel_vel)}"
                )

            if real_time and not args.headless:
                _sleep_with_window_poll(step_sleep_s)

        if args.hold_open and simulation_app.is_running():
            print("Hold-open active. Press Ctrl+C to close Isaac Sim.")
            while simulation_app.is_running():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user, closing debug session.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
