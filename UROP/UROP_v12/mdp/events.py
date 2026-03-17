#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v12/mdp/events.py]
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .observations import quat_apply, quat_conj
from .rewards import _update_hold_latch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = (0.32, 0.24, 0.24)


def _get_stage(env) -> int:
    """Read current stage from curriculum schedule."""
    p = env.cfg.curriculum.stage_schedule.params
    forced = int(p.get("eval_stage", -1))
    if forced >= 0:
        return forced

    step = int(env.common_step_counter)
    nsteps = int(p["num_steps_per_env"])
    
    # 누적 스텝(Iterations * num_steps_per_env) 계산
    s0_limit = int(p["stage0_iters"]) * nsteps
    s1_limit = s0_limit + int(p["stage1_iters"]) * nsteps
    s2_limit = s1_limit + int(p["stage2_iters"]) * nsteps

    if step < s0_limit:
        return 0 # Stage 0: 그냥 가만히 서 있기 (박스 안 던짐)
    elif step < s1_limit:
        return 1 # Stage 1: 초근접 건네주기
    elif step < s2_limit:
        return 2 # Stage 2: 약하게 던지기
    else:
        return 3 # Stage 3: 세게 던지기 (최종 난이도)


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
    stage1: dict,
    stage2: dict,
    stage3: dict,
    throw_prob_stage1: float = 1.0,
    throw_prob_stage2: float = 0.9,
    throw_prob_stage3: float = 0.9,
    max_throws_per_episode: int = 1,
) -> None:
    _ensure_urop_buffers(env)

    if max_throws_per_episode <= 0:
        return

    can = env._urop_toss_done[env_ids] < int(max_throws_per_episode)
    if not torch.any(can):
        return
    env_ids = env_ids[can]

    s = _get_stage(env)
    
    # 🔥 Stage 0이면 아무것도 하지 않고 리턴 (박스는 주차장에 그대로 있음) 🔥
    if s == 0:
        return

    # 스테이지별 설정 할당
    if s == 1:
        cfg = stage1
        prob = float(throw_prob_stage1)
    elif s == 2:
        cfg = stage2
        prob = float(throw_prob_stage2)
    else:
        cfg = stage3
        prob = float(throw_prob_stage3)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = int(env_ids.shape[0])

    # 확률적 건너뛰기 로직
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


# =========================
# Catch success bank export
# =========================
import os
from pathlib import Path


def _ensure_bank_export_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device

    if not hasattr(env, "_urop_bank_last_save_step"):
        env._urop_bank_last_save_step = torch.full((n,), -10_000_000, dtype=torch.int64, device=d)

    if not hasattr(env, "_urop_bank_saved_count"):
        env._urop_bank_saved_count = torch.zeros(n, dtype=torch.int32, device=d)

    if not hasattr(env, "_urop_bank_total_saved"):
        env._urop_bank_total_saved = 0

    if not hasattr(env, "_urop_bank_mem"):
        env._urop_bank_mem = []


def _body_name_to_idx_from_robot(robot):
    names = list(robot.data.body_names)
    return {name: i for i, name in enumerate(names)}


def _sensor_force_mag(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    s = env.scene[sensor_name]
    f = s.data.net_forces_w.reshape(env.num_envs, -1)
    return torch.norm(f, dim=-1)


def _max_force(env: "ManagerBasedRLEnv", sensor_names: list[str]) -> torch.Tensor:
    vals = [_sensor_force_mag(env, n) for n in sensor_names]
    return torch.stack(vals, dim=-1).max(dim=-1).values


def _catch_success_export_mask(
    env: "ManagerBasedRLEnv",
    min_hold_steps: int = 12,
    min_obj_z: float = 0.42,
    max_torso_obj_dist: float = 0.55,
    max_rel_speed: float = 0.55,
    min_left_force: float = 3.0,
    min_right_force: float = 3.0,
    max_anchor_drift: float = 0.18,
    max_upright_tilt_deg: float = 35.0,
) -> torch.Tensor:
    _ensure_urop_buffers(env)
    _update_hold_latch(env)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    body_map = _body_name_to_idx_from_robot(robot)
    torso_idx = body_map["torso_link"]

    torso_pos = robot.data.body_pos_w[:, torso_idx, :]
    torso_vel = robot.data.body_lin_vel_w[:, torso_idx, :]

    obj_pos = obj.data.root_pos_w
    obj_vel = obj.data.root_lin_vel_w

    dist = torch.norm(obj_pos - torso_pos, dim=-1)
    rel_speed = torch.norm(obj_vel - torso_vel, dim=-1)

    left_sensors = [
        "contact_l_shoulder_yaw", "contact_l_elbow",
        "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand"
    ]
    right_sensors = [
        "contact_r_shoulder_yaw", "contact_r_elbow",
        "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand"
    ]
    lf = _max_force(env, left_sensors)
    rf = _max_force(env, right_sensors)

    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_hold_anchor_xy, dim=-1)

    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_apply(quat_conj(q), g_world)
    upright_cos = -g_b[:, 2]
    min_upright_cos = torch.cos(torch.tensor(max_upright_tilt_deg * 3.14159265 / 180.0, device=env.device))

    mask = (
        env._urop_hold_latched
        & (env._urop_hold_steps >= int(min_hold_steps))
        & (obj_pos[:, 2] > min_obj_z)
        & (dist < max_torso_obj_dist)
        & (rel_speed < max_rel_speed)
        & (lf > min_left_force)
        & (rf > min_right_force)
        & (drift < max_anchor_drift)
        & (upright_cos > min_upright_cos)
    )
    return mask


def export_catch_success_bank(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    bank_path: str,
    min_hold_steps: int = 12,
    min_gap_steps: int = 20,
    max_total_states: int = 30000,
    flush_every: int = 256,
) -> None:
    """Collect stable post-catch states into a .pt bank file.

    Recommended usage:
      - register as an interval event in UROP_v12/env_cfg.py
      - run catch policy in eval/play
      - states will be appended in memory and flushed to disk periodically
    """
    _ensure_urop_buffers(env)
    _ensure_bank_export_buffers(env)

    if int(env_ids.numel()) == 0:
        return

    mask_all = _catch_success_export_mask(
        env,
        min_hold_steps=min_hold_steps,
    )

    if not torch.any(mask_all[env_ids]):
        return

    step_now = torch.full((env.num_envs,), int(env.common_step_counter), dtype=torch.int64, device=env.device)
    gap_ok = (step_now - env._urop_bank_last_save_step) >= int(min_gap_steps)

    final_mask = mask_all & gap_ok
    final_mask = final_mask & torch.isin(torch.arange(env.num_envs, device=env.device), env_ids)

    chosen = torch.nonzero(final_mask, as_tuple=False).squeeze(-1)
    if chosen.numel() == 0:
        return

    robot = env.scene["robot"]
    obj = env.scene["object"]

    env_origins = env.scene.env_origins[chosen]
    root_pos_local = robot.data.root_pos_w[chosen] - env_origins
    obj_pos_local = obj.data.root_pos_w[chosen] - env_origins

    root_pose = torch.cat([root_pos_local, robot.data.root_quat_w[chosen]], dim=-1)
    root_vel = torch.cat([robot.data.root_lin_vel_w[chosen], robot.data.root_ang_vel_w[chosen]], dim=-1)

    obj_pose = torch.cat([obj_pos_local, obj.data.root_quat_w[chosen]], dim=-1)
    obj_vel = torch.cat([obj.data.root_lin_vel_w[chosen], obj.data.root_ang_vel_w[chosen]], dim=-1)

    joint_pos = robot.data.joint_pos[chosen].clone()
    joint_vel = robot.data.joint_vel[chosen].clone()
    

    meta = {
        "hold_steps": env._urop_hold_steps[chosen].clone().cpu(),
        "saved_step": step_now[chosen].clone().cpu(),
        "source_env_id": chosen.clone().cpu(),
        "env_origin": env_origins.clone().cpu(),
    }

    batch = {
        "root_pose": root_pose.detach().cpu(),
        "root_vel": root_vel.detach().cpu(),
        "joint_pos": joint_pos.detach().cpu(),
        "joint_vel": joint_vel.detach().cpu(),
        "object_pose": obj_pose.detach().cpu(),
        "object_vel": obj_vel.detach().cpu(),
        "meta": meta,
    }

    env._urop_bank_mem.append(batch)
    env._urop_bank_last_save_step[chosen] = step_now[chosen]
    env._urop_bank_saved_count[chosen] += 1
    env._urop_bank_total_saved += int(chosen.numel())

    if (len(env._urop_bank_mem) >= int(flush_every)) or (env._urop_bank_total_saved >= int(max_total_states)):
        bank_path = str(bank_path)
        Path(os.path.dirname(bank_path)).mkdir(parents=True, exist_ok=True)

        # 1) current in-memory batch -> packed tensors
        new_pack = {
            "root_pose": torch.cat([x["root_pose"] for x in env._urop_bank_mem], dim=0)
            if len(env._urop_bank_mem) > 0 else torch.empty(0, 7),
            "root_vel": torch.cat([x["root_vel"] for x in env._urop_bank_mem], dim=0)
            if len(env._urop_bank_mem) > 0 else torch.empty(0, 6),
            "joint_pos": torch.cat([x["joint_pos"] for x in env._urop_bank_mem], dim=0)
            if len(env._urop_bank_mem) > 0 else torch.empty(0, robot.data.joint_pos.shape[1]),
            "joint_vel": torch.cat([x["joint_vel"] for x in env._urop_bank_mem], dim=0)
            if len(env._urop_bank_mem) > 0 else torch.empty(0, robot.data.joint_vel.shape[1]),
            "object_pose": torch.cat([x["object_pose"] for x in env._urop_bank_mem], dim=0)
            if len(env._urop_bank_mem) > 0 else torch.empty(0, 7),
            "object_vel": torch.cat([x["object_vel"] for x in env._urop_bank_mem], dim=0)
            if len(env._urop_bank_mem) > 0 else torch.empty(0, 6),
            "meta": {
                "hold_steps": torch.cat([x["meta"]["hold_steps"] for x in env._urop_bank_mem], dim=0)
                if len(env._urop_bank_mem) > 0 else torch.empty(0),
                "saved_step": torch.cat([x["meta"]["saved_step"] for x in env._urop_bank_mem], dim=0)
                if len(env._urop_bank_mem) > 0 else torch.empty(0),
            },
        }

        # 2) merge with existing packed file (if any)
        if os.path.exists(bank_path):
            old = torch.load(bank_path, map_location="cpu")

            # backward-compatible: if someone saved lists, pack them first
            def _to_tensor(v, empty_shape):
                if isinstance(v, list):
                    return torch.cat(v, dim=0) if len(v) > 0 else torch.empty(*empty_shape)
                return v

            old_root_pose = _to_tensor(old["root_pose"], (0, 7))
            old_root_vel = _to_tensor(old["root_vel"], (0, 6))
            old_joint_pos = _to_tensor(old["joint_pos"], (0, robot.data.joint_pos.shape[1]))
            old_joint_vel = _to_tensor(old["joint_vel"], (0, robot.data.joint_vel.shape[1]))
            old_object_pose = _to_tensor(old["object_pose"], (0, 7))
            old_object_vel = _to_tensor(old["object_vel"], (0, 6))

            old_hold_steps = _to_tensor(old["meta"]["hold_steps"], (0,))
            old_saved_step = _to_tensor(old["meta"]["saved_step"], (0,))

            packed = {
                "root_pose": torch.cat([old_root_pose, new_pack["root_pose"]], dim=0),
                "root_vel": torch.cat([old_root_vel, new_pack["root_vel"]], dim=0),
                "joint_pos": torch.cat([old_joint_pos, new_pack["joint_pos"]], dim=0),
                "joint_vel": torch.cat([old_joint_vel, new_pack["joint_vel"]], dim=0),
                "object_pose": torch.cat([old_object_pose, new_pack["object_pose"]], dim=0),
                "object_vel": torch.cat([old_object_vel, new_pack["object_vel"]], dim=0),
                "meta": {
                    "hold_steps": torch.cat([old_hold_steps, new_pack["meta"]["hold_steps"]], dim=0),
                    "saved_step": torch.cat([old_saved_step, new_pack["meta"]["saved_step"]], dim=0),
                },
            }
        else:
            packed = new_pack

        # 3) truncate to max_total_states
        if packed["root_pose"].shape[0] > int(max_total_states):
            keep = int(max_total_states)
            for k in ["root_pose", "root_vel", "joint_pos", "joint_vel", "object_pose", "object_vel"]:
                packed[k] = packed[k][-keep:]
            for k in ["hold_steps", "saved_step"]:
                packed["meta"][k] = packed["meta"][k][-keep:]

        # 4) save and clear memory
        torch.save(packed, bank_path)
        env._urop_bank_mem.clear()