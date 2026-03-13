"""
Build a carry reset bank from successful catch rollouts.

Usage example (run this in IsaacLab root terminal):

./isaaclab.sh -p UROP/UROP_carry_v2/tools/build_carry_state_bank.py \
    --task_module UROP.UROP_v12 \
    --task Isaac-Urop-v12-Play \
    --policy /home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v12/.../exported/policy.pt \
    --num_envs 64 \
    --num_states 4096 \
    --out /home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v2/carry_state_bank.pt \
    --headless
"""

import argparse
import importlib
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build carry reset state bank from successful catch policy rollouts.")
parser.add_argument("--task_module", type=str, default="", help="Python module to import so the gym task gets registered.")
parser.add_argument("--task", type=str, required=True, help="Catch task gym id, usually the Play task.")
parser.add_argument("--policy", type=str, required=True, help="Path to exported TorchScript policy.pt from catch training.")
parser.add_argument("--out", type=str, required=True, help="Output .pt path for the carry state bank.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_states", type=int, default=4096)
parser.add_argument("--min_hold_steps", type=int, default=12)
parser.add_argument("--max_object_speed", type=float, default=0.75)
parser.add_argument("--max_base_speed", type=float, default=0.55)
parser.add_argument("--save_every", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch


def _extract_policy_obs(obs):
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


def main():
    if args_cli.task_module:
        importlib.import_module(args_cli.task_module)

    policy_path = Path(args_cli.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    env = env.unwrapped if hasattr(env, "unwrapped") else env

    device = getattr(env, "device", "cuda:0")
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()

    obs = _extract_policy_obs(_reset_env(env))
    bank = {
        "root_pose": [],
        "root_vel": [],
        "joint_pos": [],
        "joint_vel": [],
        "object_pose": [],
        "object_vel": [],
        "obj_rel_root": [],
        "box_size": [],
    }

    step_counter = 0
    while len(bank["root_pose"]) < args_cli.num_states and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = _step_env(env, actions)
        obs = _extract_policy_obs(obs)
        step_counter += 1

        if step_counter % args_cli.save_every != 0:
            continue

        hold = getattr(env, "_urop_hold_latched", None)
        hold_steps = getattr(env, "_urop_hold_steps", None)
        if hold is None or hold_steps is None:
            raise RuntimeError("Catch env does not expose _urop_hold_latched / _urop_hold_steps. Adjust this script to your catch task.")

        robot = env.scene["robot"]
        obj = env.scene["object"]
        origins = env.scene.env_origins

        obj_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
        base_speed = torch.norm(robot.data.root_lin_vel_w[:, 0:2], dim=-1)
        mask = hold & (hold_steps >= args_cli.min_hold_steps) & (obj_speed <= args_cli.max_object_speed) & (base_speed <= args_cli.max_base_speed)

        good_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if good_ids.numel() == 0:
            continue

        root_pose = torch.cat([
            robot.data.root_pos_w[good_ids] - origins[good_ids],
            robot.data.root_quat_w[good_ids],
        ], dim=-1)
        root_vel = torch.cat([
            robot.data.root_lin_vel_w[good_ids],
            robot.data.root_ang_vel_w[good_ids],
        ], dim=-1)
        object_pose = torch.cat([
            obj.data.root_pos_w[good_ids] - origins[good_ids],
            obj.data.root_quat_w[good_ids],
        ], dim=-1)
        object_vel = torch.cat([
            obj.data.root_lin_vel_w[good_ids],
            obj.data.root_ang_vel_w[good_ids],
        ], dim=-1)
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

        if hasattr(env, "_urop_box_size"):
            bank["box_size"].append(env._urop_box_size[good_ids].cpu())

        total = sum(x.shape[0] for x in bank["root_pose"])
        print(f"[state-bank] collected {total}/{args_cli.num_states}")

    if len(bank["root_pose"]) == 0:
        raise RuntimeError("No successful carry-start states were collected. Loosen thresholds or verify the catch policy.")

    packed = {}
    for k, v in bank.items():
        if len(v) == 0:
            continue
        packed[k] = torch.cat(v, dim=0)[: args_cli.num_states]

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, out_path)
    print(f"Saved carry state bank to: {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
