from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .observations import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = (0.32, 0.24, 0.24)


def _get_stage(env) -> int:
    """Read current stage from curriculum schedule.

    Stage is used *only* to pick toss difficulty (easy -> hard).
    Rewards/terminations should not rely on stage (they rely on toss/hold signals).
    """
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
    """Extract yaw-only quaternion (w,x,y,z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _ensure_urop_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    if not hasattr(env, "_urop_toss_done"):
        env._urop_toss_done = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_toss_active"):
        env._urop_toss_active = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((n, 2), device=d)
    if not hasattr(env, "_urop_ready_joint_pos"):
        robot = env.scene["robot"]
        env._urop_ready_joint_pos = robot.data.joint_pos.clone()
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d)

    # object domain randomization buffers (used in obs/reward; keep on env.device)
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
    """Apply mass/material to PhysX (best-effort, no spam errors).

    IMPORTANT:
      PhysX tensor setters (setMasses/setMaterialProperties) require *CPU indices*.
      If you pass CUDA indices, you get the red spam:
        'expected device -1, received device 0'
    """
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

    # 1) Mass -----------------------------------------------------------------
    try:
        if hasattr(view, "get_masses") and hasattr(view, "set_masses"):
            masses = view.get_masses().clone()
            if getattr(masses, 'is_cuda', False):
                masses = masses.cpu()
            # common shape: (num_envs, 1)
            masses[env_ids_cpu, 0] = mass_cpu
            try:
                view.set_masses(masses, indices=env_ids_cpu)
            except TypeError:
                view.set_masses(masses)
    except Exception:
        pass

    # 2) Material --------------------------------------------------------------
    try:
        if hasattr(view, "get_material_properties") and hasattr(view, "set_material_properties"):
            mats = view.get_material_properties().clone()
            if getattr(mats, 'is_cuda', False):
                mats = mats.cpu()
            # common layout: mats[env, shape, {static, dynamic, restitution}]
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


def randomize_receive_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass_range=(2.0, 4.5),
    friction_range=(0.55, 0.95),
    restitution_range=(0.00, 0.06),
    size_jitter=(0.95, 1.05),
    apply_physx: bool = True,
) -> None:
    """Domain randomization for receive policy.

    - We always store sampled params into env buffers (used by obs/reward).
    - We *optionally* apply mass/material into PhysX (reset-time only).
      This avoids overfitting to one mass/friction, without spamming red errors.
    """
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    mass = torch.empty((n, 1), device=d).uniform_(*mass_range)
    friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)

    s = torch.empty((n, 3), device=d).uniform_(*size_jitter)
    base = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    size = base * s

    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = size

    if apply_physx:
        _apply_physx_mass_material_best_effort(env, env_ids, mass, friction, restitution)


def reset_object_parked(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, park: dict) -> None:
    """Reset: park the object away so the policy cannot 'pre-cheat' contact."""
    _ensure_urop_buffers(env)

    # episode state ------------------------------------------------------------
    env._urop_toss_done[env_ids] = 0
    env._urop_toss_active[env_ids] = False
    env._urop_hold_latched[env_ids] = False
    env._urop_hold_steps[env_ids] = 0
    env._urop_hold_anchor_xy[env_ids] = 0.0

    robot = env.scene["robot"]
    env._urop_spawn_xy[env_ids] = robot.data.root_pos_w[env_ids, 0:2]
    env._urop_ready_joint_pos[env_ids] = robot.data.joint_pos[env_ids]

    # randomize object parameters on reset (safe) ------------------------------
    randomize_receive_object(env, env_ids, apply_physx=True)

    # park object --------------------------------------------------------------
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = int(env_ids.shape[0])

    rp = robot.data.root_pos_w[env_ids]
    rq = _yaw_quat(robot.data.root_quat_w[env_ids])

    px = torch.empty(n, device=device).uniform_(*park["pos_x"])
    py = torch.empty(n, device=device).uniform_(*park["pos_y"])
    pz = torch.empty(n, device=device).uniform_(*park["pos_z"])
    rel_p = torch.stack([px, py, pz], dim=-1)

    p_w = rp + quat_apply(rq, rel_p)
    q_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)

    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


def toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
    throw_prob_stage1: float = 1.0,
    throw_prob_stage2: float = 0.9,
    max_throws_per_episode: int = 1,
) -> None:
    """Interval: throw/handover once per episode.

    stage0: very easy handover (near, slow) -> learn 'hug/absorb' + 'don't step'
    stage1: gentle throw
    stage2: faster throw, slightly more lateral variation
    """
    _ensure_urop_buffers(env)

    if max_throws_per_episode <= 0:
        return

    can = env._urop_toss_done[env_ids] < int(max_throws_per_episode)
    if not torch.any(can):
        return
    env_ids = env_ids[can]

    s = _get_stage(env)
    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)
    prob = 1.0 if s == 0 else float(throw_prob_stage1 if s == 1 else throw_prob_stage2)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = int(env_ids.shape[0])

    # optional stochastic skipping on harder stages
    if prob < 1.0:
        u = torch.rand(n, device=device)
        do_throw = u < prob
        ids_throw = env_ids[do_throw]
        ids_skip = env_ids[~do_throw]
        if ids_skip.numel() > 0:
            env._urop_toss_done[ids_skip] = int(max_throws_per_episode)
            env._urop_toss_active[ids_skip] = False
        if ids_throw.numel() == 0:
            return
    else:
        ids_throw = env_ids

    n2 = int(ids_throw.shape[0])
    rp = robot.data.root_pos_w[ids_throw]
    rq = _yaw_quat(robot.data.root_quat_w[ids_throw])

    px = torch.empty(n2, device=device).uniform_(*cfg["pos_x"])
    py = torch.empty(n2, device=device).uniform_(*cfg["pos_y"])
    pz = torch.empty(n2, device=device).uniform_(*cfg["pos_z"])
    rel_p = torch.stack([px, py, pz], dim=-1)

    vx = torch.empty(n2, device=device).uniform_(*cfg["vel_x"])
    vy = torch.empty(n2, device=device).uniform_(*cfg["vel_y"])
    vz = torch.empty(n2, device=device).uniform_(*cfg["vel_z"])
    rel_v = torch.stack([vx, vy, vz], dim=-1)

    p_w = rp + quat_apply(rq, rel_p)
    v_w = robot.data.root_lin_vel_w[ids_throw] + quat_apply(rq, rel_v)

    q_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n2, 1)
    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=ids_throw)

    vel6 = torch.zeros((n2, 6), device=device)
    vel6[:, 0:3] = v_w
    obj.write_root_velocity_to_sim(vel6, env_ids=ids_throw)

    env._urop_toss_done[ids_throw] += 1
    env._urop_toss_active[ids_throw] = True