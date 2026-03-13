#[/home/idim5080-2/mdj/myIsaacLabstudy/UROP/UROP_carry_v0/mdp/events.py]
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = (0.32, 0.24, 0.24)


def _get_stage(env) -> int:
    p = env.cfg.curriculum.stage_schedule.params
    forced = int(p.get("eval_stage", -1))
    if forced >= 0:
        return forced

    step = int(env.common_step_counter)
    nsteps = int(p["num_steps_per_env"])

    s0_limit = int(p["stage0_iters"]) * nsteps
    s1_limit = s0_limit + int(p["stage1_iters"]) * nsteps
    s2_limit = s1_limit + int(p["stage2_iters"]) * nsteps

    if step < s0_limit:
        return 0
    elif step < s1_limit:
        return 1
    elif step < s2_limit:
        return 2
    else:
        return 3


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
        dim=-1,
    )


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return _quat_from_yaw(yaw)


def _ensure_urop_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device

    if not hasattr(env, "_urop_command"):
        env._urop_command = torch.zeros((n, 3), device=d)

    if not hasattr(env, "_urop_carry_rel_target"):
        env._urop_carry_rel_target = torch.tensor([0.38, 0.0, 0.34], device=d).unsqueeze(0).repeat(n, 1)

    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), 3.0, device=d)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.7, device=d)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.02, device=d)


def _apply_physx_mass_material_best_effort(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass: torch.Tensor,
    friction: torch.Tensor,
    restitution: torch.Tensor,
) -> None:
    obj = env.scene["object"]
    try:
        view = obj.root_physx_view
    except Exception:
        return

    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    mass_cpu = mass.detach().to(device="cpu").squeeze(-1)
    fric_cpu = friction.detach().to(device="cpu").squeeze(-1)
    rest_cpu = restitution.detach().to(device="cpu").squeeze(-1)

    try:
        if hasattr(view, "get_masses") and hasattr(view, "set_masses"):
            masses = view.get_masses().clone()
            if getattr(masses, "is_cuda", False):
                masses = masses.cpu()
            masses[env_ids_cpu, 0] = mass_cpu
            try:
                view.set_masses(masses, indices=env_ids_cpu)
            except TypeError:
                view.set_masses(masses)
    except Exception:
        pass

    try:
        if hasattr(view, "get_material_properties") and hasattr(view, "set_material_properties"):
            mats = view.get_material_properties().clone()
            if getattr(mats, "is_cuda", False):
                mats = mats.cpu()
            fr = fric_cpu.view(-1, 1)
            rs = rest_cpu.view(-1, 1)
            mats[env_ids_cpu, :, 0] = fr
            mats[env_ids_cpu, :, 1] = 0.85 * fr
            mats[env_ids_cpu, :, 2] = rs
            try:
                view.set_material_properties(mats, indices=env_ids_cpu)
            except TypeError:
                view.set_material_properties(mats)
    except Exception:
        pass


def randomize_carry_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass_range=(2.0, 5.0),
    friction_range=(0.55, 0.95),
    restitution_range=(0.00, 0.05),
    size_jitter=(0.95, 1.08),
    apply_physx: bool = True,
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    mass = torch.empty((n, 1), device=d).uniform_(*mass_range)
    friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)

    scale = torch.empty((n, 3), device=d).uniform_(*size_jitter)
    base = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    size = base * scale

    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = size

    if apply_physx:
        _apply_physx_mass_material_best_effort(env, env_ids, mass, friction, restitution)


def reset_object_in_carry_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
    stage3: dict,
    apply_physx: bool = True,
) -> None:
    """Reset the object already in a pre-grasp / pre-embrace pose.

    This is the key difference from catch: phase-3 begins after successful receive.
    """
    _ensure_urop_buffers(env)
    s = _get_stage(env)

    if s == 0:
        cfg = stage0
    elif s == 1:
        cfg = stage1
    elif s == 2:
        cfg = stage2
    else:
        cfg = stage3

    randomize_carry_object(
        env,
        env_ids,
        mass_range=cfg["mass_range"],
        friction_range=cfg["friction_range"],
        restitution_range=cfg["restitution_range"],
        size_jitter=cfg["size_jitter"],
        apply_physx=apply_physx,
    )

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = int(env_ids.shape[0])

    root_pos = robot.data.root_pos_w[env_ids]
    root_yaw_q = _yaw_quat(robot.data.root_quat_w[env_ids])

    px = torch.empty(n, device=device).uniform_(*cfg["hold_x"])
    py = torch.empty(n, device=device).uniform_(*cfg["hold_y"])
    pz = torch.empty(n, device=device).uniform_(*cfg["hold_z"])
    rel_target = torch.stack([px, py, pz], dim=-1)
    env._urop_carry_rel_target[env_ids] = rel_target

    pos_w = root_pos + quat_apply(root_yaw_q, rel_target)

    yaw_noise = torch.empty(n, device=device).uniform_(*cfg["box_yaw_deg"])
    yaw_noise = yaw_noise * math.pi / 180.0
    obj_q = _quat_from_yaw(yaw_noise)
    obj_pose = torch.cat([pos_w, obj_q], dim=-1)

    obj.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


def resample_carry_command(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
    stage3: dict,
) -> None:
    _ensure_urop_buffers(env)
    s = _get_stage(env)

    if s == 0:
        cfg = stage0
    elif s == 1:
        cfg = stage1
    elif s == 2:
        cfg = stage2
    else:
        cfg = stage3

    n = int(env_ids.shape[0])
    d = env.device

    vx = torch.empty(n, device=d).uniform_(*cfg["lin_vel_x"])
    vy = torch.empty(n, device=d).uniform_(*cfg["lin_vel_y"])
    wz = torch.empty(n, device=d).uniform_(*cfg["ang_vel_z"])

    env._urop_command[env_ids, 0] = vx
    env._urop_command[env_ids, 1] = vy
    env._urop_command[env_ids, 2] = wz
