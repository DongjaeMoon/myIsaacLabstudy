#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/mdp/events.py]
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_stage(env) -> int:
    p = env.cfg.curriculum.stage_schedule.params
    forced = int(p.get("eval_stage", -1))
    if forced in (0, 1, 2):
        return forced
    step = int(env.common_step_counter)
    stage0 = int(p["stage0_iters"])
    stage1 = int(p["stage1_iters"])
    nsteps = int(p["num_steps_per_env"])
    s1 = stage0 * nsteps
    s2 = (stage0 + stage1) * nsteps
    if step < s1:
        return 0
    elif step < s2:
        return 1
    else:
        return 2

def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    # extract yaw-only quaternion from (w,x,y,z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)

def reset_robot_base_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
):
    s = _get_stage(env)
    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)

    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    n = env_ids.shape[0]

    vel = robot.data.root_vel_w[env_ids].clone()

    vel[:, 0] = torch.empty(n, device=device).uniform_(*cfg["lin_x"])
    vel[:, 1] = torch.empty(n, device=device).uniform_(*cfg["lin_y"])
    vel[:, 2] = 0.0
    vel[:, 3] = 0.0
    vel[:, 4] = 0.0
    vel[:, 5] = torch.empty(n, device=device).uniform_(*cfg["yaw_rate"])

    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)


def reset_and_toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
):
    s = _get_stage(env)
    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)

    robot = env.scene["robot"]
    obj   = env.scene["object"]

    device = obj.data.root_pos_w.device
    n = env_ids.shape[0]

    rp = robot.data.root_pos_w[env_ids]
    rq = _yaw_quat(robot.data.root_quat_w[env_ids])

    px = torch.empty(n, device=device).uniform_(*cfg["pos_x"])
    py = torch.empty(n, device=device).uniform_(*cfg["pos_y"])
    pz = torch.empty(n, device=device).uniform_(*cfg["pos_z"])
    rel_p = torch.stack([px, py, pz], dim=-1)

    vx = torch.empty(n, device=device).uniform_(*cfg["vel_x"])
    vy = torch.empty(n, device=device).uniform_(*cfg["vel_y"])
    vz = torch.empty(n, device=device).uniform_(*cfg["vel_z"])
    rel_v = torch.stack([vx, vy, vz], dim=-1)

    p_w = rp + quat_apply(rq, rel_p)
    v_w = robot.data.root_lin_vel_w[env_ids] + quat_apply(rq, rel_v)

    q_w = torch.tensor([1,0,0,0], device=device).repeat(n,1)

    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=env_ids)
    vel6 = torch.zeros((n,6), device=device)
    vel6[:,0:3] = v_w
    obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)

