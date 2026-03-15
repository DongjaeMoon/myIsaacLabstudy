"""
Build a carry reset bank from successful catch rollouts.

Run this in IsaacLab root terminal, e.g.

Smoke test:
./isaaclab.sh -p UROP/UROP_carry_v2/tools/build_carry_state_bank.py \
    --task_module UROP.UROP_v12 \
    --task Isaac-Urop-v12-Play \
    --policy /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt \
    --num_envs 16 \
    --num_states 64 \
    --min_hold_steps 3 \
    --max_object_speed 2.0 \
    --max_base_speed 2.0 \
    --save_every 1 \
    --debug_every 100 \
    --force_eval_stage 3 \
    --force_throw_prob_stage3 1.0 \
    --out /home/idim5080-2/mdj/myIsaacLabstudy/UROP/UROP_carry_v2/carry_state_bank_test.pt \
    --headless

Full bank:
./isaaclab.sh -p UROP/UROP_carry_v2/tools/build_carry_state_bank.py \
    --task_module UROP.UROP_v12 \
    --task Isaac-Urop-v12-Play \
    --policy /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt \
    --num_envs 64 \
    --num_states 4096 \
    --min_hold_steps 6 \
    --max_object_speed 1.2 \
    --max_base_speed 0.9 \
    --save_every 2 \
    --debug_every 200 \
    --force_eval_stage 3 \
    --force_throw_prob_stage3 1.0 \
    --out /home/idim5080-2/mdj/myIsaacLabstudy/UROP/UROP_carry_v2/carry_state_bank.pt \
    --headless
"""

import argparse
import importlib
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Make project root importable so that:
#   import UROP.UROP_v12
# works even when this script is executed as a file path.
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]   # .../myIsaacLabstudy
_UROP_ROOT = _THIS_FILE.parents[2]      # .../myIsaacLabstudy/UROP

for p in [str(_PROJECT_ROOT), str(_UROP_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Build carry reset state bank from successful catch policy rollouts."
)
parser.add_argument(
    "--task_module",
    type=str,
    default="",
    help="Python module to import so the gym task gets registered. Example: UROP.UROP_v12",
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Catch task gym id, usually the Play task. Example: Isaac-Urop-v12-Play",
)
parser.add_argument(
    "--policy",
    type=str,
    required=True,
    help="Path to exported TorchScript policy.pt from catch training.",
)
parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="Output .pt path for the carry state bank.",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_states", type=int, default=4096)

# Collection thresholds
parser.add_argument("--min_hold_steps", type=int, default=6)
parser.add_argument("--max_object_speed", type=float, default=1.2)
parser.add_argument("--max_base_speed", type=float, default=0.9)
parser.add_argument("--save_every", type=int, default=2)
parser.add_argument("--debug_every", type=int, default=200)
parser.add_argument("--max_total_steps", type=int, default=200000)

# Optional cfg overrides for Play env
parser.add_argument(
    "--force_eval_stage",
    type=int,
    default=-1,
    help="If >=0, override env_cfg.curriculum.stage_schedule.params['eval_stage']",
)
parser.add_argument(
    "--force_throw_prob_stage3",
    type=float,
    default=-1.0,
    help="If >=0, override env_cfg.events.toss.params['throw_prob_stage3']",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

try:
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:
    from omni.isaac.lab_tasks.utils import parse_env_cfg


def _extract_policy_obs(obs):
    """Extract policy obs tensor from Gym/Gymnasium/IsaacLab style outputs."""
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        return next(iter(obs.values()))
    if isinstance(obs, tuple):
        return _extract_policy_obs(obs[0])
    return obs


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, actions):
    out = env.step(actions)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = terminated | truncated
        return obs, rew, done, info
    if len(out) == 4:
        obs, rew, done, info = out
        return obs, rew, done, info
    raise RuntimeError(f"Unexpected env.step output length: {len(out)}")


def _safe_import_task_module(task_module: str):
    if not task_module:
        return
    try:
        importlib.import_module(task_module)
        print(f"[state-bank] imported task module: {task_module}")
    except ModuleNotFoundError:
        short_name = task_module.split(".")[-1]
        print(f"[state-bank] failed to import '{task_module}', trying '{short_name}' instead...")
        importlib.import_module(short_name)
        print(f"[state-bank] imported task module: {short_name}")


def _get_hold_buffers(env):
    """Try several likely attribute names for hold latch and hold duration."""
    hold_name_candidates = [
        "_urop_hold_latched",
        "urop_hold_latched",
        "_hold_latched",
        "hold_latched",
    ]
    hold_steps_candidates = [
        "_urop_hold_steps",
        "urop_hold_steps",
        "_hold_steps",
        "hold_steps",
    ]

    hold = None
    hold_steps = None

    for name in hold_name_candidates:
        if hasattr(env, name):
            hold = getattr(env, name)
            break

    for name in hold_steps_candidates:
        if hasattr(env, name):
            hold_steps = getattr(env, name)
            break

    if hold is None or hold_steps is None:
        related = [k for k in dir(env) if ("hold" in k.lower() or "urop" in k.lower())]
        raise RuntimeError(
            "Could not find hold buffers in catch env.\n"
            f"Expected one of hold-latched names: {hold_name_candidates}\n"
            f"Expected one of hold-steps names: {hold_steps_candidates}\n"
            f"Related env attrs: {related[:80]}"
        )

    if not torch.is_tensor(hold) or not torch.is_tensor(hold_steps):
        raise RuntimeError(
            f"Hold buffers found, but they are not tensors. "
            f"type(hold)={type(hold)}, type(hold_steps)={type(hold_steps)}"
        )

    return hold, hold_steps


def _maybe_override_env_cfg(env_cfg):
    """Optional safety override so the builder definitely uses hard catch-eval settings."""
    # Force eval stage if requested
    if args_cli.force_eval_stage >= 0:
        try:
            env_cfg.curriculum.stage_schedule.params["eval_stage"] = int(args_cli.force_eval_stage)
            print(f"[state-bank] force eval_stage = {args_cli.force_eval_stage}")
        except Exception as e:
            print(f"[state-bank] warning: failed to override eval_stage: {e}")

    # Force throw prob if requested
    if args_cli.force_throw_prob_stage3 >= 0.0:
        try:
            env_cfg.events.toss.params["throw_prob_stage3"] = float(args_cli.force_throw_prob_stage3)
            print(f"[state-bank] force throw_prob_stage3 = {args_cli.force_throw_prob_stage3}")
        except Exception as e:
            print(f"[state-bank] warning: failed to override throw_prob_stage3: {e}")

    # Print current values if available
    try:
        eval_stage = env_cfg.curriculum.stage_schedule.params.get("eval_stage", None)
    except Exception:
        eval_stage = None
    try:
        throw_prob = env_cfg.events.toss.params.get("throw_prob_stage3", None)
    except Exception:
        throw_prob = None

    print(f"[state-bank] env_cfg summary: eval_stage={eval_stage}, throw_prob_stage3={throw_prob}")
    return env_cfg


def main():
    _safe_import_task_module(args_cli.task_module)

    policy_path = Path(args_cli.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env_cfg = _maybe_override_env_cfg(env_cfg)

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = env.unwrapped if hasattr(env, "unwrapped") else env

        device = getattr(env, "device", args_cli.device)
        print(f"[state-bank] env device = {device}")

        policy = torch.jit.load(str(policy_path), map_location=device)
        policy.eval()
        print(f"[state-bank] loaded policy: {policy_path}")

        obs = _extract_policy_obs(_reset_env(env))
        if torch.is_tensor(obs):
            obs = obs.to(device)

        bank = {
            "root_pose": [],
            "root_vel": [],
            "joint_pos": [],
            "joint_vel": [],
            "object_pose": [],
            "object_vel": [],
            "obj_rel_root": [],
        }

        # Optional object-size field if your env exposes it
        maybe_box_size_key = None
        for k in ["_urop_box_size", "urop_box_size", "_box_size", "box_size"]:
            if hasattr(env, k):
                maybe_box_size_key = k
                bank["box_size"] = []
                break

        robot = env.scene["robot"]
        obj = env.scene["object"]

        step_counter = 0
        total_collected = 0

        while total_collected < args_cli.num_states and simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)

            obs, _, _, _ = _step_env(env, actions)
            obs = _extract_policy_obs(obs)
            if torch.is_tensor(obs):
                obs = obs.to(device)

            step_counter += 1

            # Periodic debug print
            if step_counter % args_cli.debug_every == 0:
                hold, hold_steps = _get_hold_buffers(env)
                obj_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
                base_speed = torch.norm(robot.data.root_lin_vel_w[:, 0:2], dim=-1)

                num_hold = int(hold.sum().item())
                num_hold_long = int((hold & (hold_steps >= args_cli.min_hold_steps)).sum().item())
                num_obj_ok = int((obj_speed <= args_cli.max_object_speed).sum().item())
                num_base_ok = int((base_speed <= args_cli.max_base_speed).sum().item())

                print(
                    f"[state-bank] step={step_counter} "
                    f"hold={num_hold}/{args_cli.num_envs} "
                    f"hold_long={num_hold_long}/{args_cli.num_envs} "
                    f"obj_ok={num_obj_ok}/{args_cli.num_envs} "
                    f"base_ok={num_base_ok}/{args_cli.num_envs} "
                    f"obj_speed_mean={obj_speed.mean().item():.3f} "
                    f"base_speed_mean={base_speed.mean().item():.3f}"
                )

            # Don't check every step if user wants sparser capture
            if step_counter % args_cli.save_every != 0:
                if step_counter >= args_cli.max_total_steps:
                    break
                continue

            hold, hold_steps = _get_hold_buffers(env)
            origins = env.scene.env_origins

            obj_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
            base_speed = torch.norm(robot.data.root_lin_vel_w[:, 0:2], dim=-1)

            mask = (
                hold
                & (hold_steps >= args_cli.min_hold_steps)
                & (obj_speed <= args_cli.max_object_speed)
                & (base_speed <= args_cli.max_base_speed)
            )

            good_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if good_ids.numel() == 0:
                if step_counter >= args_cli.max_total_steps:
                    break
                continue

            root_pose = torch.cat(
                [
                    robot.data.root_pos_w[good_ids] - origins[good_ids],
                    robot.data.root_quat_w[good_ids],
                ],
                dim=-1,
            )
            root_vel = torch.cat(
                [
                    robot.data.root_lin_vel_w[good_ids],
                    robot.data.root_ang_vel_w[good_ids],
                ],
                dim=-1,
            )
            object_pose = torch.cat(
                [
                    obj.data.root_pos_w[good_ids] - origins[good_ids],
                    obj.data.root_quat_w[good_ids],
                ],
                dim=-1,
            )
            object_vel = torch.cat(
                [
                    obj.data.root_lin_vel_w[good_ids],
                    obj.data.root_ang_vel_w[good_ids],
                ],
                dim=-1,
            )
            joint_pos = robot.data.joint_pos[good_ids]
            joint_vel = robot.data.joint_vel[good_ids]
            obj_rel_root = obj.data.root_pos_w[good_ids] - robot.data.root_pos_w[good_ids]

            bank["root_pose"].append(root_pose.cpu())
            bank["root_vel"].append(root_vel.cpu())
            bank["joint_pos"].append(joint_pos.cpu())
            bank["joint_vel"].append(joint_vel.cpu())
            bank["object_pose"].append(object_pose.cpu())
            bank["object_vel"].append(object_vel.cpu())
            bank["obj_rel_root"].append(obj_rel_root.cpu())

            if maybe_box_size_key is not None:
                bank["box_size"].append(getattr(env, maybe_box_size_key)[good_ids].cpu())

            total_collected = sum(x.shape[0] for x in bank["root_pose"])
            print(f"[state-bank] collected {total_collected}/{args_cli.num_states}")

            if step_counter >= args_cli.max_total_steps:
                break

        if len(bank["root_pose"]) == 0:
            raise RuntimeError(
                "No carry-start states were collected.\n"
                "Likely causes:\n"
                "1) hold buffers are not the names assumed by this script\n"
                "2) thresholds are too strict\n"
                "3) catch policy/env rollout is not producing stable hold states\n"
                "Try smoke-test settings first:\n"
                "  --num_envs 16 --num_states 64 --min_hold_steps 3 "
                "--max_object_speed 2.0 --max_base_speed 2.0 --save_every 1"
            )

        packed = {}
        for k, v in bank.items():
            if len(v) == 0:
                continue
            packed[k] = torch.cat(v, dim=0)[: args_cli.num_states]

        # Add metadata for later debugging / provenance
        packed["meta"] = {
            "task": args_cli.task,
            "task_module": args_cli.task_module,
            "policy": str(policy_path),
            "num_envs": args_cli.num_envs,
            "num_states_requested": args_cli.num_states,
            "num_states_saved": int(packed["root_pose"].shape[0]),
            "min_hold_steps": args_cli.min_hold_steps,
            "max_object_speed": args_cli.max_object_speed,
            "max_base_speed": args_cli.max_base_speed,
            "save_every": args_cli.save_every,
            "debug_every": args_cli.debug_every,
            "max_total_steps": args_cli.max_total_steps,
            "force_eval_stage": args_cli.force_eval_stage,
            "force_throw_prob_stage3": args_cli.force_throw_prob_stage3,
            "state_convention": "robot/object root poses are stored in local env-origin frame",
        }

        torch.save(packed, out_path)
        print(f"[state-bank] saved carry state bank to: {out_path}")
        print(f"[state-bank] final saved states: {packed['root_pose'].shape[0]}")

    finally:
        try:
            if env is not None:
                env.close()
        except Exception as e:
            print(f"[state-bank] warning during env.close(): {e}")

        try:
            simulation_app.close()
        except Exception as e:
            print(f"[state-bank] warning during simulation_app.close(): {e}")


if __name__ == "__main__":
    main()