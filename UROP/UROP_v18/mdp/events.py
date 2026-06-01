from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import get_controlled_joint_indices, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = scene_objects_cfg.OBJECT_BASE_SIZE


# Hidden delivery-family ids used only by the event generator.
# Actor never observes this; reward must remain trajectory-family independent.
DELIVERY_IDLE = 0
DELIVERY_BALLISTIC = 1
DELIVERY_PUSH = 2
DELIVERY_CARRIED = 3
DELIVERY_LOB = 4
DELIVERY_LATE_VISIBLE = 5
DELIVERY_ABORT = 6
DELIVERY_PASS_BY = 7

TOSS_FAMILIES = {
    "ballistic": DELIVERY_BALLISTIC,
    "push": DELIVERY_PUSH,
    "push_release": DELIVERY_PUSH,
    "carried": DELIVERY_CARRIED,
    "carried_release": DELIVERY_CARRIED,
    "lob": DELIVERY_LOB,
    "late_visible": DELIVERY_LATE_VISIBLE,
    "late_visible_toss": DELIVERY_LATE_VISIBLE,
}
NO_TOSS_FAMILIES = {
    "idle": DELIVERY_IDLE,
    "visible_idle": DELIVERY_IDLE,
    "abort": DELIVERY_ABORT,
    "approach_abort": DELIVERY_ABORT,
    "pass_by": DELIVERY_PASS_BY,
    "lateral_pass": DELIVERY_PASS_BY,
}
FAMILY_TO_NAME = {
    DELIVERY_IDLE: "idle",
    DELIVERY_BALLISTIC: "ballistic",
    DELIVERY_PUSH: "push_release",
    DELIVERY_CARRIED: "carried_release",
    DELIVERY_LOB: "lob",
    DELIVERY_LATE_VISIBLE: "late_visible_toss",
    DELIVERY_ABORT: "approach_abort",
    DELIVERY_PASS_BY: "pass_by",
}


def _stage_dict_value(table: dict | None, stage: int, default):
    if not table:
        return default
    return table.get(f"stage{stage}", table.get(str(stage), default))


def _normalize_prob_dict(prob_dict: dict, allowed: dict[str, int]) -> tuple[list[str], torch.Tensor]:
    names: list[str] = []
    probs: list[float] = []
    for raw_name, raw_prob in prob_dict.items():
        name = str(raw_name).lower()
        if name not in allowed:
            continue
        p = max(float(raw_prob), 0.0)
        if p <= 0.0:
            continue
        names.append(name)
        probs.append(p)
    if not names:
        # fallback is filled by caller
        names = list(allowed.keys())[:1]
        probs = [1.0]
    prob_t = torch.tensor(probs, dtype=torch.float32)
    prob_t = prob_t / torch.clamp(prob_t.sum(), min=1.0e-6)
    return names, prob_t


def _sample_delivery_families(
    env: "ManagerBasedRLEnv",
    num_samples: int,
    stage: int,
    base_should_toss: torch.Tensor,
    delivery_randomization: dict | None,
) -> torch.Tensor:
    """Sample a broad hidden delivery process.

    The policy never observes this family id.  It only sees noisy object rel pos/vel/tag_visible.
    Families are used to generate diverse trajectories: idle, pass-by, carried approach, push-release,
    late-visible tosses, and ballistic tosses.
    """
    delivery_randomization = delivery_randomization or {}
    device = env.device
    family = torch.full((num_samples,), DELIVERY_IDLE, device=device, dtype=torch.long)

    toss_table = delivery_randomization.get("toss_family_prob_by_stage", {})
    no_toss_table = delivery_randomization.get("no_toss_family_prob_by_stage", {})
    toss_probs = _stage_dict_value(
        toss_table,
        stage,
        {"ballistic": 0.35, "carried_release": 0.30, "push_release": 0.20, "lob": 0.10, "late_visible": 0.05},
    )
    no_toss_probs = _stage_dict_value(
        no_toss_table,
        stage,
        {"idle": 0.50, "pass_by": 0.30, "approach_abort": 0.20},
    )

    for want_toss, prob_dict, allowed in [
        (True, toss_probs, TOSS_FAMILIES),
        (False, no_toss_probs, NO_TOSS_FAMILIES),
    ]:
        mask = base_should_toss if want_toss else (~base_should_toss)
        if not torch.any(mask):
            continue
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        names, probs = _normalize_prob_dict(prob_dict, allowed)
        probs = probs.to(device=device)
        choices = torch.multinomial(probs, int(ids.shape[0]), replacement=True)
        mapped = torch.tensor([allowed[names[int(i)]] for i in choices.detach().cpu().tolist()], device=device, dtype=torch.long)
        family[ids] = mapped
    return family


def _delivery_family_has_release(family: torch.Tensor) -> torch.Tensor:
    return (family == DELIVERY_BALLISTIC) | (family == DELIVERY_PUSH) | (family == DELIVERY_CARRIED) | (family == DELIVERY_LOB) | (family == DELIVERY_LATE_VISIBLE)


def _sample_pose_velocity_from_cfg(
    cfg: dict,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    default_pos=((0.75, 1.55), (-0.45, 0.45), (-0.18, 0.32)),
    default_vel=((-0.02, 0.04), (-0.05, 0.05), (-0.02, 0.02)),
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.stack(
        [
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg.get("pos_x", default_pos[0])),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg.get("pos_y", default_pos[1])),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg.get("pos_z", default_pos[2])),
        ],
        dim=-1,
    )
    vel = torch.stack(
        [
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg.get("vel_x", default_vel[0])),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg.get("vel_y", default_vel[1])),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg.get("vel_z", default_vel[2])),
        ],
        dim=-1,
    )
    return pos, vel


def _get_stage(env: "ManagerBasedRLEnv") -> int:
    params = env.cfg.curriculum.stage_schedule.params
    forced = int(params.get("eval_stage", -1))
    if forced >= 0:
        return forced

    step = int(env.common_step_counter)
    nsteps = int(params["num_steps_per_env"])
    s0_limit = int(params["stage0_iters"]) * nsteps
    s1_limit = s0_limit + int(params["stage1_iters"]) * nsteps
    s2_limit = s1_limit + int(params["stage2_iters"]) * nsteps

    if step < s0_limit:
        return 0
    if step < s1_limit:
        return 1
    if step < s2_limit:
        return 2
    return 3


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


def _gravity_world(env: "ManagerBasedRLEnv", num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gravity = getattr(env.sim.cfg, "gravity", (0.0, 0.0, -9.81))
    return torch.tensor(gravity, device=device, dtype=dtype).unsqueeze(0).repeat(num_samples, 1)


def _ensure_urop_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype

    if not hasattr(env, "_urop_toss_done"):
        env._urop_toss_done = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_toss_active"):
        env._urop_toss_active = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_should_toss"):
        env._urop_should_toss = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_toss_wait_s"):
        env._urop_toss_wait_s = torch.full((n, 1), 999.0, device=d, dtype=dtype)
    if not hasattr(env, "_urop_toss_probability"):
        env._urop_toss_probability = torch.zeros((n, 1), dtype=dtype, device=d)
    if not hasattr(env, "_urop_last_toss_spawn_rel"):
        env._urop_last_toss_spawn_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_last_toss_target_rel"):
        env._urop_last_toss_target_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_last_toss_flight_time"):
        env._urop_last_toss_flight_time = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_object_scene_visible"):
        env._urop_object_scene_visible = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_visible_hold_rel"):
        env._urop_visible_hold_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_visible_drift_rel_vel"):
        env._urop_visible_drift_rel_vel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_visible_start_s"):
        env._urop_visible_start_s = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_visible_yaw"):
        env._urop_visible_yaw = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_pending_toss_valid"):
        env._urop_pending_toss_valid = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_pending_toss_spawn_rel"):
        env._urop_pending_toss_spawn_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_pending_toss_vel_rel"):
        env._urop_pending_toss_vel_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_pending_toss_target_rel"):
        env._urop_pending_toss_target_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_pending_toss_flight_time"):
        env._urop_pending_toss_flight_time = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_pending_toss_quat"):
        env._urop_pending_toss_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d, dtype=dtype).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_pending_toss_ang_vel"):
        env._urop_pending_toss_ang_vel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((n, 2), device=d, dtype=dtype)
    if not hasattr(env, "_urop_spawn_yaw"):
        env._urop_spawn_yaw = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_ready_joint_pos"):
        robot = env.scene["robot"]
        env._urop_ready_joint_pos = robot.data.default_joint_pos.clone()
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d, dtype=dtype)
    if not hasattr(env, "_urop_no_toss_episode_count"):
        env._urop_no_toss_episode_count = torch.zeros((), dtype=torch.int64, device=d)
    if not hasattr(env, "_urop_toss_episode_count"):
        env._urop_toss_episode_count = torch.zeros((), dtype=torch.int64, device=d)
    if not hasattr(env, "_urop_delivery_family"):
        env._urop_delivery_family = torch.zeros(n, dtype=torch.long, device=d)
    if not hasattr(env, "_urop_delivery_release_due"):
        env._urop_delivery_release_due = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_toss_active_start_step"):
        env._urop_toss_active_start_step = torch.zeros(n, dtype=torch.long, device=d)

    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d, dtype=dtype).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), scene_objects_cfg.OBJECT_DEFAULT_MASS, device=d, dtype=dtype)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.8, device=d, dtype=dtype)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.02, device=d, dtype=dtype)
    if not hasattr(env, "_urop_robot_contact_friction"):
        env._urop_robot_contact_friction = torch.full((n, 1), 0.85, device=d, dtype=dtype)
    if not hasattr(env, "_urop_floor_friction"):
        env._urop_floor_friction = torch.full((n, 1), 0.90, device=d, dtype=dtype)

    if not hasattr(env, "_urop_projected_gravity_noise_std"):
        env._urop_projected_gravity_noise_std = torch.full((n, 1), 0.015, device=d, dtype=dtype)
    if not hasattr(env, "_urop_base_ang_vel_noise_std"):
        env._urop_base_ang_vel_noise_std = torch.full((n, 1), 0.04, device=d, dtype=dtype)
    if not hasattr(env, "_urop_joint_pos_noise_std"):
        env._urop_joint_pos_noise_std = torch.full((n, 1), 0.010, device=d, dtype=dtype)
    if not hasattr(env, "_urop_joint_vel_noise_std"):
        env._urop_joint_vel_noise_std = torch.full((n, 1), 0.08, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_filter_pos"):
        env._urop_obj_filter_pos = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_filter_vel"):
        env._urop_obj_filter_vel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos"):
        env._urop_obj_obs_pos = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel"):
        env._urop_obj_obs_vel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_visible"):
        env._urop_obj_visible = torch.zeros((n, 1), device=d, dtype=torch.bool)
    if not hasattr(env, "_urop_obj_obs_alpha"):
        env._urop_obj_obs_alpha = torch.full((n, 1), 0.65, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_drop_prob"):
        env._urop_obj_obs_drop_prob = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos_noise_std"):
        env._urop_obj_obs_pos_noise_std = torch.full((n, 1), 0.01, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel_noise_std"):
        env._urop_obj_obs_vel_noise_std = torch.full((n, 1), 0.05, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_cache_global_step"):
        env._urop_obj_obs_cache_global_step = -1
    if not hasattr(env, "_urop_obj_obs_cache_episode_len"):
        env._urop_obj_obs_cache_episode_len = torch.full((n,), -1, device=d, dtype=torch.long)
    if not hasattr(env, "_urop_obj_obs_pos_bias"):
        env._urop_obj_obs_pos_bias = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel_bias"):
        env._urop_obj_obs_vel_bias = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos_scale"):
        env._urop_obj_obs_pos_scale = torch.ones((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel_scale"):
        env._urop_obj_obs_vel_scale = torch.ones((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_noise_spike_prob"):
        env._urop_obj_obs_noise_spike_prob = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_noise_spike_scale"):
        env._urop_obj_obs_noise_spike_scale = torch.full((n, 1), 4.0, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_false_positive_prob"):
        env._urop_obj_obs_false_positive_prob = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_tag_visible_noise_std"):
        env._urop_tag_visible_noise_std = torch.full((n, 1), 0.02, device=d, dtype=dtype)
    if not hasattr(env, "_urop_prev_action_obs_noise_std"):
        env._urop_prev_action_obs_noise_std = torch.full((n, 1), 0.006, device=d, dtype=dtype)
    if not hasattr(env, "_urop_mode_noise_std"):
        env._urop_mode_noise_std = torch.full((n, 1), 0.010, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_latency_steps"):
        env._urop_obj_latency_steps = torch.zeros((n, 1), device=d, dtype=torch.long)


def _apply_physx_mass_best_effort(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass: torch.Tensor,
) -> None:
    obj = env.scene["object"]
    try:
        view = obj.root_physx_view
    except Exception:
        return

    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    try:
        masses = view.get_masses().clone()
        if getattr(masses, "is_cuda", False):
            masses = masses.cpu()
        masses[env_ids_cpu, 0] = mass.detach().to(device="cpu").squeeze(-1)
        try:
            view.set_masses(masses, indices=env_ids_cpu)
        except TypeError:
            view.set_masses(masses)
    except Exception:
        pass


def _apply_physx_material_best_effort(
    view,
    env_ids: torch.Tensor,
    friction: torch.Tensor,
    restitution: torch.Tensor,
) -> None:
    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    try:
        mats = view.get_material_properties().clone()
        if getattr(mats, "is_cuda", False):
            mats = mats.cpu()
        fr = friction.detach().to(device="cpu").squeeze(-1).view(-1, 1)
        rs = restitution.detach().to(device="cpu").squeeze(-1).view(-1, 1)
        mats[env_ids_cpu, :, 0] = fr
        mats[env_ids_cpu, :, 1] = 0.92 * fr
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
    mass_range=(1.0, 6.0),
    friction_range=(0.25, 1.50),
    restitution_range=(0.00, 0.18),
    size_scale_range=(0.80, 1.25),
    apply_physx: bool = True,
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    mass = torch.empty((n, 1), device=d).uniform_(*mass_range)
    friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)
    scale = torch.empty((n, 3), device=d).uniform_(*size_scale_range)
    size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1) * scale

    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = size

    if apply_physx:
        _apply_physx_mass_best_effort(env, env_ids, mass)
        try:
            _apply_physx_material_best_effort(env.scene["object"].root_physx_view, env_ids, friction, restitution)
        except Exception:
            pass


def randomize_robot_contact_material(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    friction_range=(0.40, 1.30),
    restitution_range=(0.00, 0.04),
    floor_friction_range=(0.35, 1.40),
    apply_physx: bool = True,
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    robot_friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    robot_restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)
    floor_friction = torch.empty((n, 1), device=d).uniform_(*floor_friction_range)
    # The ground plane is shared across environments, so we fold the sampled floor friction
    # into the robot-side material to preserve per-env contact variation without global side effects.
    effective_robot_friction = torch.clamp(robot_friction * (floor_friction / 0.90), 0.15, 1.80)

    env._urop_robot_contact_friction[env_ids] = effective_robot_friction
    env._urop_floor_friction[env_ids] = floor_friction

    if apply_physx:
        try:
            _apply_physx_material_best_effort(
                env.scene["robot"].root_physx_view, env_ids, effective_robot_friction, robot_restitution
            )
        except Exception:
            pass
    # Ground is shared across envs in the default scene, so we keep a best-effort sampled buffer here.


def _uniform_vec(env: "ManagerBasedRLEnv", num_envs: int, ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> torch.Tensor:
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype
    return torch.stack(
        [
            torch.empty(num_envs, device=d, dtype=dtype).uniform_(*ranges[0]),
            torch.empty(num_envs, device=d, dtype=dtype).uniform_(*ranges[1]),
            torch.empty(num_envs, device=d, dtype=dtype).uniform_(*ranges[2]),
        ],
        dim=-1,
    )


def randomize_object_observation(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    projected_gravity_noise_std_range=(0.008, 0.045),
    base_ang_vel_noise_std_range=(0.02, 0.12),
    joint_pos_noise_std_range=(0.004, 0.030),
    joint_vel_noise_std_range=(0.04, 0.28),
    prev_action_noise_std_range=(0.002, 0.020),
    mode_noise_std_range=(0.004, 0.030),
    obj_pos_noise_range=(0.010, 0.095),
    obj_vel_noise_range=(0.06, 0.55),
    obj_pos_bias_range=(-0.035, 0.035),
    obj_vel_bias_range=(-0.10, 0.10),
    obj_pos_scale_range=(0.88, 1.12),
    obj_vel_scale_range=(0.75, 1.25),
    drop_prob_range=(0.03, 0.38),
    false_positive_prob_range=(0.0, 0.015),
    tag_visible_noise_std_range=(0.010, 0.050),
    alpha_range=(0.15, 0.85),
    latency_steps_range=(0, 5),
    noise_spike_prob_range=(0.00, 0.08),
    noise_spike_scale_range=(2.5, 7.0),
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    # Every actor-observation group gets randomized noise in v18.  The critic stays clean.
    env._urop_projected_gravity_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(
        *projected_gravity_noise_std_range
    )
    env._urop_base_ang_vel_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(
        *base_ang_vel_noise_std_range
    )
    env._urop_joint_pos_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*joint_pos_noise_std_range)
    env._urop_joint_vel_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*joint_vel_noise_std_range)
    env._urop_prev_action_obs_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*prev_action_noise_std_range)
    env._urop_mode_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*mode_noise_std_range)

    env._urop_obj_obs_pos_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*obj_pos_noise_range)
    env._urop_obj_obs_vel_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*obj_vel_noise_range)
    env._urop_obj_obs_drop_prob[env_ids] = torch.empty((n, 1), device=d).uniform_(*drop_prob_range)
    env._urop_obj_obs_alpha[env_ids] = torch.empty((n, 1), device=d).uniform_(*alpha_range)

    bias_ranges_p = (obj_pos_bias_range, obj_pos_bias_range, obj_pos_bias_range)
    bias_ranges_v = (obj_vel_bias_range, obj_vel_bias_range, obj_vel_bias_range)
    env._urop_obj_obs_pos_bias[env_ids] = _uniform_vec(env, n, bias_ranges_p)
    env._urop_obj_obs_vel_bias[env_ids] = _uniform_vec(env, n, bias_ranges_v)
    env._urop_obj_obs_pos_scale[env_ids] = torch.empty((n, 3), device=d).uniform_(*obj_pos_scale_range)
    env._urop_obj_obs_vel_scale[env_ids] = torch.empty((n, 3), device=d).uniform_(*obj_vel_scale_range)
    env._urop_obj_obs_false_positive_prob[env_ids] = torch.empty((n, 1), device=d).uniform_(*false_positive_prob_range)
    env._urop_tag_visible_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*tag_visible_noise_std_range)
    env._urop_obj_obs_noise_spike_prob[env_ids] = torch.empty((n, 1), device=d).uniform_(*noise_spike_prob_range)
    env._urop_obj_obs_noise_spike_scale[env_ids] = torch.empty((n, 1), device=d).uniform_(*noise_spike_scale_range)

    lo, hi = int(latency_steps_range[0]), int(latency_steps_range[1])
    lo = max(0, lo)
    hi = max(lo, min(8, hi))
    env._urop_obj_latency_steps[env_ids] = torch.randint(lo, hi + 1, (n, 1), device=d, dtype=torch.long)

def _sample_wait_times(env: "ManagerBasedRLEnv", num_envs: int, stage: int, wait_time_ranges: dict) -> torch.Tensor:
    d = env.device
    key = f"stage{stage}"
    if stage <= 0 or key not in wait_time_ranges:
        return torch.full((num_envs, 1), 999.0, device=d, dtype=env.scene["robot"].data.root_pos_w.dtype)
    low, high = wait_time_ranges[key]
    return torch.empty((num_envs, 1), device=d, dtype=env.scene["robot"].data.root_pos_w.dtype).uniform_(low, high)


def _sample_should_toss(
    env: "ManagerBasedRLEnv",
    num_envs: int,
    stage: int,
    toss_probability_by_stage: dict | None,
) -> tuple[torch.Tensor, float]:
    toss_probability_by_stage = toss_probability_by_stage or {}
    prob = float(toss_probability_by_stage.get(f"stage{stage}", 0.0))
    prob = max(0.0, min(1.0, prob))

    should_toss = torch.rand(num_envs, device=env.device) < prob
    if prob <= 0.0:
        should_toss.zero_()
    elif prob >= 1.0:
        should_toss.fill_(True)
    return should_toss, prob


def _sample_joint_noise(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    joint_noise: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_noise = joint_noise or {}
    pos_ranges = {
        "lower_body": tuple(joint_noise.get("lower_body_pos", (-0.03, 0.03))),
        "waist": tuple(joint_noise.get("waist_pos", (-0.02, 0.02))),
        "arm": tuple(joint_noise.get("arm_pos", (-0.05, 0.05))),
        "wrist": tuple(joint_noise.get("wrist_pos", (-0.03, 0.03))),
    }
    vel_range = tuple(joint_noise.get("velocity", (-0.08, 0.08)))

    n = int(env_ids.shape[0])
    d = env.device
    pos_min = torch.empty((scene_objects_cfg.EXPECTED_ACTION_DIM,), device=d)
    pos_max = torch.empty((scene_objects_cfg.EXPECTED_ACTION_DIM,), device=d)

    for index, name in enumerate(scene_objects_cfg.CONTROLLED_JOINT_NAMES):
        if "wrist" in name:
            rng = pos_ranges["wrist"]
        elif "shoulder" in name or "elbow" in name:
            rng = pos_ranges["arm"]
        elif "waist" in name:
            rng = pos_ranges["waist"]
        else:
            rng = pos_ranges["lower_body"]
        pos_min[index] = rng[0]
        pos_max[index] = rng[1]

    pos_noise = torch.rand((n, scene_objects_cfg.EXPECTED_ACTION_DIM), device=d) * (pos_max - pos_min) + pos_min
    vel_noise = torch.empty((n, scene_objects_cfg.EXPECTED_ACTION_DIM), device=d).uniform_(*vel_range)
    return pos_noise, vel_noise


def reset_autonomous_episode(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    park: dict,
    wait_time_ranges: dict,
    toss_probability_by_stage: dict | None = None,
    joint_noise: dict | None = None,
    root_xy_range=(-0.02, 0.02),
    root_yaw_range=(-0.05, 0.05),
    object_randomization: dict | None = None,
    robot_material_randomization: dict | None = None,
    floor_material_randomization: dict | None = None,
    observation_randomization: dict | None = None,
    visibility_randomization: dict | None = None,
    delivery_randomization: dict | None = None,
) -> None:
    _ensure_urop_buffers(env)

    object_randomization = object_randomization or {}
    robot_material_randomization = robot_material_randomization or {}
    floor_material_randomization = floor_material_randomization or {}
    observation_randomization = observation_randomization or {}
    visibility_randomization = visibility_randomization or {}
    delivery_randomization = delivery_randomization or {}

    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    n = int(env_ids.shape[0])

    stage = _get_stage(env)
    base_should_toss, toss_probability = _sample_should_toss(env, n, stage, toss_probability_by_stage)
    delivery_family = _sample_delivery_families(env, n, stage, base_should_toss, delivery_randomization)
    should_toss = _delivery_family_has_release(delivery_family)
    wait_times = _sample_wait_times(env, n, stage, wait_time_ranges)
    no_toss_wait = torch.full_like(wait_times, 999.0)
    wait_times = torch.where(should_toss.unsqueeze(-1), wait_times, no_toss_wait)

    env._urop_toss_done[env_ids] = 0
    env._urop_toss_active[env_ids] = False
    env._urop_toss_active_start_step[env_ids] = 0
    env._urop_should_toss[env_ids] = should_toss
    env._urop_delivery_family[env_ids] = delivery_family
    env._urop_delivery_release_due[env_ids] = should_toss
    env._urop_toss_wait_s[env_ids] = wait_times
    env._urop_toss_probability[env_ids] = toss_probability
    env._urop_last_toss_spawn_rel[env_ids] = 0.0
    env._urop_last_toss_target_rel[env_ids] = 0.0
    env._urop_last_toss_flight_time[env_ids] = 0.0
    env._urop_hold_latched[env_ids] = False
    env._urop_hold_steps[env_ids] = 0
    env._urop_hold_anchor_xy[env_ids] = 0.0
    if hasattr(env, "_urop_hold_cache_global_step"):
        env._urop_hold_cache_global_step = -1
    if hasattr(env, "_urop_hold_cache_episode_len"):
        env._urop_hold_cache_episode_len[env_ids] = -1
    env._urop_obj_filter_pos[env_ids] = 0.0
    env._urop_obj_filter_vel[env_ids] = 0.0
    env._urop_obj_obs_pos[env_ids] = 0.0
    env._urop_obj_obs_vel[env_ids] = 0.0
    env._urop_obj_visible[env_ids] = False
    env._urop_object_scene_visible[env_ids] = False
    env._urop_visible_hold_rel[env_ids] = 0.0
    env._urop_visible_drift_rel_vel[env_ids] = 0.0
    env._urop_visible_start_s[env_ids] = 0.0
    env._urop_visible_yaw[env_ids] = 0.0
    env._urop_pending_toss_valid[env_ids] = False
    env._urop_pending_toss_spawn_rel[env_ids] = 0.0
    env._urop_pending_toss_vel_rel[env_ids] = 0.0
    env._urop_pending_toss_target_rel[env_ids] = 0.0
    env._urop_pending_toss_flight_time[env_ids] = 0.0
    env._urop_pending_toss_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d).repeat(n, 1)
    env._urop_pending_toss_ang_vel[env_ids] = 0.0
    if hasattr(env, "_urop_obj_latency_buffer_pos"):
        env._urop_obj_latency_buffer_pos[env_ids] = 0.0
    if hasattr(env, "_urop_obj_latency_buffer_vel"):
        env._urop_obj_latency_buffer_vel[env_ids] = 0.0
    if hasattr(env, "_urop_obj_latency_buffer_visible"):
        env._urop_obj_latency_buffer_visible[env_ids] = False
    env._urop_obj_obs_cache_global_step = -1
    env._urop_obj_obs_cache_episode_len[env_ids] = -1

    env._urop_ready_joint_pos[env_ids] = robot.data.default_joint_pos[env_ids].clone()

    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    xy_noise = torch.empty((n, 2), device=d).uniform_(*root_xy_range)
    yaw_noise = torch.empty((n,), device=d).uniform_(*root_yaw_range)
    yaw_quat = _quat_from_euler_xyz(
        torch.zeros(n, device=d),
        torch.zeros(n, device=d),
        yaw_noise,
    )

    root_pos = default_root_state[:, 0:3]
    root_pos[:, 0:2] += xy_noise
    root_quat = quat_mul(default_root_state[:, 3:7], yaw_quat)
    root_vel = torch.zeros((n, 6), device=d)

    robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    controlled_ids = get_controlled_joint_indices(env)
    pos_noise, vel_noise = _sample_joint_noise(env, env_ids, joint_noise)
    joint_pos[:, controlled_ids] += pos_noise
    joint_vel[:, controlled_ids] = vel_noise

    joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = robot.data.soft_joint_vel_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=env_ids)
    env._urop_spawn_xy[env_ids] = root_pos[:, 0:2]
    env._urop_spawn_yaw[env_ids, 0] = torch.atan2(
        2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
        1.0 - 2.0 * (root_quat[:, 2] ** 2 + root_quat[:, 3] ** 2),
    )
    env._urop_toss_episode_count += should_toss.to(dtype=torch.int64).sum()
    env._urop_no_toss_episode_count += (~should_toss).to(dtype=torch.int64).sum()

    randomize_receive_object(env, env_ids, **object_randomization)
    randomize_robot_contact_material(
        env,
        env_ids,
        friction_range=robot_material_randomization.get("friction_range", (0.70, 1.00)),
        restitution_range=robot_material_randomization.get("restitution_range", (0.00, 0.02)),
        floor_friction_range=floor_material_randomization.get("friction_range", (0.75, 1.05)),
        apply_physx=robot_material_randomization.get("apply_physx", True),
    )
    randomize_object_observation(env, env_ids, **observation_randomization)

    # ------------------------------------------------------------
    # v18.2 broad stochastic delivery model.
    # The hidden family samples how the object moves, but the actor never sees it.
    # tag_visible remains a noisy perception signal; receiving is learned from rel_pos/rel_vel.
    # ------------------------------------------------------------
    toss_params = getattr(env.cfg.events, "toss").params if hasattr(env.cfg.events, "toss") else {}
    cfg, _ = _select_toss_cfg(
        stage,
        toss_params.get("stage1", {}),
        toss_params.get("stage2", {}),
        toss_params.get("stage3", {}),
        toss_params.get("throw_prob_stage1", 1.0),
        toss_params.get("throw_prob_stage2", 1.0),
        toss_params.get("throw_prob_stage3", 1.0),
    )

    visible_prob_by_stage = visibility_randomization.get(
        "pre_toss_visible_probability_by_stage",
        {"stage0": 0.0, "stage1": 0.70, "stage2": 0.82, "stage3": 0.90},
    )
    idle_visible_prob_by_stage = visibility_randomization.get(
        "idle_visible_probability_by_stage",
        {"stage0": 0.90, "stage1": 0.45, "stage2": 0.35, "stage3": 0.28},
    )
    pre_visible_prob = float(visible_prob_by_stage.get(f"stage{stage}", 0.80))
    idle_visible_prob = float(idle_visible_prob_by_stage.get(f"stage{stage}", 0.35))

    scene_visible = torch.zeros(n, dtype=torch.bool, device=d)
    late_visible = delivery_family == DELIVERY_LATE_VISIBLE
    toss_scene_visible = should_toss & (~late_visible) & (torch.rand(n, device=d) < pre_visible_prob)
    idle_scene_visible = (~should_toss) & (torch.rand(n, device=d) < idle_visible_prob)
    # pass-by / abort families are intentionally visible often: they teach "do not hug just because it is visible".
    forced_no_toss_visible = (~should_toss) & ((delivery_family == DELIVERY_PASS_BY) | (delivery_family == DELIVERY_ABORT))
    scene_visible |= toss_scene_visible | idle_scene_visible | forced_no_toss_visible

    # Pending release state is sampled at reset so pre-release carried motion and release match.
    if torch.any(should_toss):
        toss_local = should_toss.nonzero(as_tuple=False).squeeze(-1)
        nt = int(toss_local.shape[0])
        fam_toss = delivery_family[toss_local]
        rel_p_toss, rel_v_toss, target_rel, flight_time, quat_pending, ang_vel_pending = _sample_release_states_for_families(
            env, cfg, fam_toss, nt, d, obj.data.root_pos_w.dtype
        )
        ids_toss = env_ids[toss_local]
        env._urop_pending_toss_valid[ids_toss] = True
        env._urop_pending_toss_spawn_rel[ids_toss] = rel_p_toss
        env._urop_pending_toss_vel_rel[ids_toss] = rel_v_toss
        env._urop_pending_toss_target_rel[ids_toss] = target_rel
        env._urop_pending_toss_flight_time[ids_toss] = flight_time
        env._urop_pending_toss_quat[ids_toss] = quat_pending
        env._urop_pending_toss_ang_vel[ids_toss] = ang_vel_pending

    # Visible/no-release states: static idle, lateral pass-by, or abort/recede.  These are not
    # privileged labels; they just fill the rel_pos/rel_vel state space with non-catch cases.
    idle_cfg = visibility_randomization.get(
        "idle_visible_pose",
        {
            "pos_x": (0.75, 1.75),
            "pos_y": (-0.60, 0.60),
            "pos_z": (-0.25, 0.35),
            "vel_x": (-0.03, 0.06),
            "vel_y": (-0.06, 0.06),
            "vel_z": (-0.02, 0.02),
        },
    )
    idle_rel, idle_vel = _sample_pose_velocity_from_cfg(idle_cfg, n, d, obj.data.root_pos_w.dtype)

    abort_cfg = delivery_randomization.get(
        "abort_visible_pose",
        {
            "pos_x": (0.65, 1.65),
            "pos_y": (-0.60, 0.60),
            "pos_z": (-0.20, 0.36),
            "vel_x": (-0.08, 0.08),
            "vel_y": (-0.08, 0.08),
            "vel_z": (-0.03, 0.03),
        },
    )
    abort_rel, abort_vel = _sample_pose_velocity_from_cfg(abort_cfg, n, d, obj.data.root_pos_w.dtype)
    pass_cfg = delivery_randomization.get(
        "pass_by_visible_pose",
        {
            "pos_x": (0.55, 1.40),
            "pos_y": (-0.90, 0.90),
            "pos_z": (-0.18, 0.36),
            "vel_x": (-0.04, 0.08),
            "vel_y": (-0.55, 0.55),
            "vel_z": (-0.03, 0.03),
        },
    )
    pass_rel, pass_vel = _sample_pose_velocity_from_cfg(pass_cfg, n, d, obj.data.root_pos_w.dtype)
    pass_vel[:, 1] = torch.where(torch.abs(pass_vel[:, 1]) < 0.12, 0.12 * torch.sign(pass_vel[:, 1] + 1.0e-6), pass_vel[:, 1])

    no_toss_rel = idle_rel
    no_toss_vel = idle_vel
    abort_mask = delivery_family == DELIVERY_ABORT
    pass_mask = delivery_family == DELIVERY_PASS_BY
    no_toss_rel = torch.where(abort_mask.unsqueeze(-1), abort_rel, no_toss_rel)
    no_toss_vel = torch.where(abort_mask.unsqueeze(-1), abort_vel, no_toss_vel)
    no_toss_rel = torch.where(pass_mask.unsqueeze(-1), pass_rel, no_toss_rel)
    no_toss_vel = torch.where(pass_mask.unsqueeze(-1), pass_vel, no_toss_vel)

    release_rel = env._urop_pending_toss_spawn_rel[env_ids]
    hold_rel = torch.where(should_toss.unsqueeze(-1), release_rel, no_toss_rel)
    hold_vel = torch.where(should_toss.unsqueeze(-1), torch.zeros_like(no_toss_vel), no_toss_vel)

    # For many visible release episodes, the object starts farther away and is carried toward
    # the sampled release state.  This prevents overfitting to "object appears exactly at release".
    start_cfg = delivery_randomization.get(
        "pre_release_start_pose",
        {
            "pos_x": (0.85, 2.05),
            "pos_y": (-0.75, 0.75),
            "pos_z": (-0.28, 0.46),
            "vel_x": (0.0, 0.0),
            "vel_y": (0.0, 0.0),
            "vel_z": (0.0, 0.0),
        },
    )
    start_rel, _ = _sample_pose_velocity_from_cfg(start_cfg, n, d, obj.data.root_pos_w.dtype)
    start_prob_by_stage = delivery_randomization.get(
        "pre_release_start_probability_by_stage",
        {"stage0": 0.0, "stage1": 0.45, "stage2": 0.65, "stage3": 0.78},
    )
    start_prob = float(start_prob_by_stage.get(f"stage{stage}", 0.65))
    use_carried_start = should_toss & scene_visible & (torch.rand(n, device=d) < start_prob)
    hold_rel = torch.where(use_carried_start.unsqueeze(-1), start_rel, hold_rel)
    wait_safe = torch.clamp(wait_times, min=0.25, max=8.0)
    approach_vel = (release_rel - hold_rel) / wait_safe
    approach_vel = torch.clamp(approach_vel, -1.20, 1.20)
    hold_vel = torch.where(use_carried_start.unsqueeze(-1), approach_vel, hold_vel)

    env._urop_visible_hold_rel[env_ids] = hold_rel
    env._urop_visible_drift_rel_vel[env_ids] = hold_vel
    env._urop_object_scene_visible[env_ids] = scene_visible
    env._urop_visible_start_s[env_ids] = torch.empty((n, 1), device=d).uniform_(
        *visibility_randomization.get("visibility_start_s", (0.0, 0.45))
    )
    env._urop_visible_yaw[env_ids] = torch.empty((n, 1), device=d).uniform_(-0.55, 0.55)

    # Initial object placement. Hidden/no-tag objects are parked out of view; visible objects
    # are held in the front scene by update_visible_object_before_toss().
    rel_p = torch.stack(
        [
            torch.empty(n, device=d).uniform_(*park["pos_x"]),
            torch.empty(n, device=d).uniform_(*park["pos_y"]),
            torch.empty(n, device=d).uniform_(*park["pos_z"]),
        ],
        dim=-1,
    )
    rel_p = torch.where(scene_visible.unsqueeze(-1), hold_rel, rel_p)
    pos_w = root_pos + quat_apply(_yaw_quat(root_quat), rel_p)
    quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d).repeat(n, 1)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=d), env_ids=env_ids)


def update_visible_object_before_toss(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    max_hold_speed: float = 0.25,
    jitter_amp: float = 0.015,
) -> None:
    """Script an invisible human-held object before release.

    This creates the critical training case: tag_visible=1 with a real object_rel_pos,
    but no catch is required yet. The object is held or moved slowly until the toss event
    releases it dynamically. No actor observation dimension is added.
    """
    _ensure_urop_buffers(env)
    if env_ids.numel() == 0:
        return

    inactive = ~env._urop_toss_active[env_ids]
    visible = env._urop_object_scene_visible[env_ids]
    mask = inactive & visible
    if not torch.any(mask):
        return

    ids = env_ids[mask]
    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    n = int(ids.shape[0])
    dtype = obj.data.root_pos_w.dtype

    t = env.episode_length_buf[ids].float().unsqueeze(-1) * float(env.step_dt)
    started = t >= env._urop_visible_start_s[ids]

    root_pos = robot.data.root_pos_w[ids]
    root_yaw = _yaw_quat(robot.data.root_quat_w[ids])
    dt_since_start = torch.clamp(t - env._urop_visible_start_s[ids], min=0.0)

    drift = torch.clamp(env._urop_visible_drift_rel_vel[ids], -float(max_hold_speed), float(max_hold_speed)) * dt_since_start
    # Tiny periodic hand tremor to avoid a perfectly static visual object.
    phase = 1.7 * t + env._urop_visible_yaw[ids]
    jitter = torch.cat(
        [
            torch.zeros((n, 1), device=d, dtype=dtype),
            torch.sin(phase) * float(jitter_amp),
            torch.cos(1.3 * phase) * float(jitter_amp),
        ],
        dim=-1,
    )
    rel = env._urop_visible_hold_rel[ids] + drift + jitter
    rel = torch.where(started, rel, torch.tensor([1.8, 0.0, -0.60], device=d, dtype=dtype).unsqueeze(0).repeat(n, 1))

    pos_w = root_pos + quat_apply(root_yaw, rel)
    quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d, dtype=dtype).repeat(n, 1)
    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=ids)

    vel6 = torch.zeros((n, 6), device=d, dtype=dtype)
    # root velocity + relative drift velocity in the yaw frame. Before visibility starts, keep zero.
    rel_vel = torch.where(started, torch.clamp(env._urop_visible_drift_rel_vel[ids], -float(max_hold_speed), float(max_hold_speed)), torch.zeros((n, 3), device=d, dtype=dtype))
    vel6[:, 0:3] = robot.data.root_lin_vel_w[ids] + quat_apply(root_yaw, rel_vel)
    obj.write_root_velocity_to_sim(vel6, env_ids=ids)


def push_robot_root_velocity(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0_xy_range=(-0.15, 0.15),
    stage1_xy_range=(-0.25, 0.25),
    stage2_xy_range=(-0.35, 0.35),
    stage3_xy_range=(-0.45, 0.45),
    stage0_yaw_range=(-0.10, 0.10),
    stage1_yaw_range=(-0.18, 0.18),
    stage2_yaw_range=(-0.25, 0.25),
    stage3_yaw_range=(-0.35, 0.35),
    z_velocity_range=(-0.02, 0.02),
    hold_xy_scale=0.70,
    hold_yaw_scale=0.65,
    max_xy_speed=1.40,
    max_yaw_speed=1.00,
) -> None:
    _ensure_urop_buffers(env)

    if env_ids.numel() == 0:
        return

    stage = _get_stage(env)
    if stage <= 0:
        xy_range = stage0_xy_range
        yaw_range = stage0_yaw_range
    elif stage == 1:
        xy_range = stage1_xy_range
        yaw_range = stage1_yaw_range
    elif stage == 2:
        xy_range = stage2_xy_range
        yaw_range = stage2_yaw_range
    else:
        xy_range = stage3_xy_range
        yaw_range = stage3_yaw_range

    robot = env.scene["robot"]
    if hasattr(robot.data, "root_vel_w"):
        root_vel = robot.data.root_vel_w[env_ids].clone()
    else:
        root_vel = torch.cat(
            [robot.data.root_lin_vel_w[env_ids].clone(), robot.data.root_ang_vel_w[env_ids].clone()],
            dim=-1,
        )

    d = env.device
    n = int(env_ids.shape[0])
    dv_xy = torch.empty((n, 2), device=d).uniform_(*xy_range)
    dv_z = torch.empty((n, 1), device=d).uniform_(*z_velocity_range)
    dyaw = torch.empty((n, 1), device=d).uniform_(*yaw_range)

    if hasattr(env, "_urop_hold_latched"):
        hold_mask = env._urop_hold_latched[env_ids].unsqueeze(-1).float()
        dv_xy = dv_xy * (1.0 - hold_mask + hold_mask * hold_xy_scale)
        dv_z = dv_z * (1.0 - hold_mask + hold_mask * hold_xy_scale)
        dyaw = dyaw * (1.0 - hold_mask + hold_mask * hold_yaw_scale)

    root_vel[:, 0:2] += dv_xy
    root_vel[:, 2:3] += dv_z
    root_vel[:, 5:6] += dyaw

    xy_speed = torch.norm(root_vel[:, 0:2], dim=-1, keepdim=True).clamp_min(1.0e-6)
    root_vel[:, 0:2] = root_vel[:, 0:2] * torch.clamp(float(max_xy_speed) / xy_speed, max=1.0)
    root_vel[:, 2:3] = torch.clamp(root_vel[:, 2:3], -0.20, 0.20)
    root_vel[:, 5:6] = torch.clamp(root_vel[:, 5:6], -float(max_yaw_speed), float(max_yaw_speed))

    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def _select_toss_cfg(stage: int, stage1: dict, stage2: dict, stage3: dict, prob_stage1: float, prob_stage2: float, prob_stage3: float) -> tuple[dict, float]:
    if stage <= 1:
        return stage1, float(prob_stage1)
    if stage == 2:
        return stage2, float(prob_stage2)
    return stage3, float(prob_stage3)


def _sample_uniform_triplet(
    cfg: dict,
    prefix: str,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg[f"{prefix}_x"]),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg[f"{prefix}_y"]),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg[f"{prefix}_z"]),
        ],
        dim=-1,
    )


def _sample_independent_velocity(
    cfg: dict,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg["vel_x"]),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg["vel_y"]),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg["vel_z"]),
        ],
        dim=-1,
    )


def _sample_targeted_ballistic_velocity(
    env: "ManagerBasedRLEnv",
    cfg: dict,
    spawn_rel: torch.Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_rel = _sample_uniform_triplet(cfg, "target", num_samples, device, dtype)
    flight_time = torch.empty((num_samples, 1), device=device, dtype=dtype).uniform_(*cfg["flight_time"])
    gravity = _gravity_world(env, num_samples, device, dtype)
    rel_vel = (target_rel - spawn_rel - 0.5 * gravity * torch.square(flight_time)) / flight_time

    # Best-effort velocity safety clamp for admissible close human tosses.
    # These limits are configured per stage in env_cfg.py.
    max_speed = cfg.get("max_speed", None)
    if max_speed is not None:
        speed = torch.norm(rel_vel, dim=-1, keepdim=True).clamp_min(1.0e-6)
        scale = torch.clamp(float(max_speed) / speed, max=1.0)
        rel_vel = rel_vel * scale

    max_vy_abs = cfg.get("max_vy_abs", None)
    if max_vy_abs is not None:
        rel_vel[:, 1] = torch.clamp(rel_vel[:, 1], -float(max_vy_abs), float(max_vy_abs))

    max_vz_abs = cfg.get("max_vz_abs", None)
    if max_vz_abs is not None:
        rel_vel[:, 2] = torch.clamp(rel_vel[:, 2], -float(max_vz_abs), float(max_vz_abs))

    return rel_vel, target_rel, flight_time


def _sample_toss_state(
    env: "ManagerBasedRLEnv",
    cfg: dict,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spawn_rel = _sample_uniform_triplet(cfg, "spawn", num_samples, device, dtype)

    sampler = str(cfg.get("sampler", "targeted_ballistic")).lower()
    if sampler == "independent":
        rel_vel = _sample_independent_velocity(cfg, num_samples, device, dtype)
        target_rel = torch.zeros_like(spawn_rel)
        flight_time = torch.zeros((num_samples, 1), device=device, dtype=dtype)
    else:
        rel_vel, target_rel, flight_time = _sample_targeted_ballistic_velocity(
            env, cfg, spawn_rel, num_samples, device, dtype
        )

    return spawn_rel, rel_vel, target_rel, flight_time



def _sample_release_states_for_families(
    env: "ManagerBasedRLEnv",
    base_cfg: dict,
    family: torch.Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample release pose/velocity for a hidden mixture of delivery families.

    All returned positions/velocities are root-body-frame relative.  The mixture is hidden from
    the actor to reduce overfitting to any one delivery grammar.
    """
    rel_p = torch.zeros((num_samples, 3), device=device, dtype=dtype)
    rel_v = torch.zeros((num_samples, 3), device=device, dtype=dtype)
    target_rel = torch.zeros((num_samples, 3), device=device, dtype=dtype)
    flight_time = torch.zeros((num_samples, 1), device=device, dtype=dtype)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).repeat(num_samples, 1)
    ang_vel = torch.zeros((num_samples, 3), device=device, dtype=dtype)

    for fam_id, fam_name in FAMILY_TO_NAME.items():
        mask = family == int(fam_id)
        if not torch.any(mask):
            continue
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        n = int(ids.shape[0])
        cfg = base_cfg.get(fam_name, base_cfg)
        # Some aliases for convenience in env_cfg.py.
        if fam_id == DELIVERY_PUSH:
            cfg = base_cfg.get("push_release", cfg)
        elif fam_id == DELIVERY_CARRIED:
            cfg = base_cfg.get("carried_release", cfg)
        elif fam_id == DELIVERY_LATE_VISIBLE:
            cfg = base_cfg.get("late_visible_toss", base_cfg.get("ballistic", base_cfg))
        elif fam_id == DELIVERY_BALLISTIC:
            cfg = base_cfg.get("ballistic", base_cfg)

        if fam_id in (DELIVERY_PUSH, DELIVERY_CARRIED):
            # Non-ballistic release: carried/pushed object already has approach velocity.
            rel_p_i = _sample_uniform_triplet(cfg, "spawn", n, device, dtype)
            rel_v_i = _sample_independent_velocity(cfg, n, device, dtype)
            target_i = torch.zeros_like(rel_p_i)
            flight_i = torch.zeros((n, 1), device=device, dtype=dtype)
        else:
            rel_p_i, rel_v_i, target_i, flight_i = _sample_toss_state(env, cfg, n, device, dtype)

        # Generic safety clamps for all families.
        max_speed = cfg.get("max_speed", None)
        if max_speed is not None:
            speed = torch.norm(rel_v_i, dim=-1, keepdim=True).clamp_min(1.0e-6)
            rel_v_i = rel_v_i * torch.clamp(float(max_speed) / speed, max=1.0)
        max_vy_abs = cfg.get("max_vy_abs", None)
        if max_vy_abs is not None:
            rel_v_i[:, 1] = torch.clamp(rel_v_i[:, 1], -float(max_vy_abs), float(max_vy_abs))
        max_vz_abs = cfg.get("max_vz_abs", None)
        if max_vz_abs is not None:
            rel_v_i[:, 2] = torch.clamp(rel_v_i[:, 2], -float(max_vz_abs), float(max_vz_abs))

        roll = torch.empty(n, device=device, dtype=dtype).uniform_(*cfg.get("roll", (-0.03, 0.03)))
        pitch = torch.empty(n, device=device, dtype=dtype).uniform_(*cfg.get("pitch", (-0.04, 0.04)))
        yaw = torch.empty(n, device=device, dtype=dtype).uniform_(*cfg.get("yaw", (-0.10, 0.10)))
        quat_i = _quat_from_euler_xyz(roll, pitch, yaw)
        ang_i = torch.stack(
            [
                torch.empty(n, device=device, dtype=dtype).uniform_(*cfg.get("ang_vel_x", (-0.12, 0.12))),
                torch.empty(n, device=device, dtype=dtype).uniform_(*cfg.get("ang_vel_y", (-0.12, 0.12))),
                torch.empty(n, device=device, dtype=dtype).uniform_(*cfg.get("ang_vel_z", (-0.20, 0.20))),
            ],
            dim=-1,
        )

        rel_p[ids] = rel_p_i
        rel_v[ids] = rel_v_i
        target_rel[ids] = target_i
        flight_time[ids] = flight_i
        quat[ids] = quat_i
        ang_vel[ids] = ang_i

    return rel_p, rel_v, target_rel, flight_time, quat, ang_vel


def toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage1: dict,
    stage2: dict,
    stage3: dict,
    throw_prob_stage1: float = 1.0,
    throw_prob_stage2: float = 1.0,
    throw_prob_stage3: float = 1.0,
    max_throws_per_episode: int = 1,
) -> None:
    _ensure_urop_buffers(env)

    if max_throws_per_episode <= 0:
        return

    stage = _get_stage(env)
    if stage == 0:
        return

    if env_ids.numel() == 0:
        return

    episode_time_s = env.episode_length_buf[env_ids].float().unsqueeze(-1) * float(env.step_dt)
    due = episode_time_s >= env._urop_toss_wait_s[env_ids]
    due &= env._urop_should_toss[env_ids].unsqueeze(-1)
    due &= (~env._urop_toss_active[env_ids]).unsqueeze(-1)
    due &= (env._urop_toss_done[env_ids] < int(max_throws_per_episode)).unsqueeze(-1)

    if not torch.any(due):
        return

    ids_due = env_ids[due.squeeze(-1)]
    cfg, throw_prob = _select_toss_cfg(
        stage,
        stage1,
        stage2,
        stage3,
        throw_prob_stage1,
        throw_prob_stage2,
        throw_prob_stage3,
    )

    if throw_prob < 1.0:
        keep = torch.rand(ids_due.shape[0], device=env.device) < throw_prob
        ids_throw = ids_due[keep]
        ids_skip = ids_due[~keep]
        if ids_skip.numel() > 0:
            env._urop_toss_done[ids_skip] = int(max_throws_per_episode)
            env._urop_toss_active[ids_skip] = False
        if ids_throw.numel() == 0:
            return
    else:
        ids_throw = ids_due

    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    n = int(ids_throw.shape[0])
    dtype = obj.data.root_pos_w.dtype

    root_pos = robot.data.root_pos_w[ids_throw]
    root_yaw = _yaw_quat(robot.data.root_quat_w[ids_throw])

    pending_valid = getattr(env, "_urop_pending_toss_valid", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))[ids_throw]
    if torch.any(pending_valid):
        rel_p = env._urop_pending_toss_spawn_rel[ids_throw].clone()
        rel_v = env._urop_pending_toss_vel_rel[ids_throw].clone()
        target_rel = env._urop_pending_toss_target_rel[ids_throw].clone()
        flight_time = env._urop_pending_toss_flight_time[ids_throw].clone()
        quat_w = env._urop_pending_toss_quat[ids_throw].clone()
        ang_vel = env._urop_pending_toss_ang_vel[ids_throw].clone()
        if not torch.all(pending_valid):
            sample_ids = (~pending_valid).nonzero(as_tuple=False).squeeze(-1)
            rel_p_s, rel_v_s, target_s, flight_s = _sample_toss_state(env, cfg, int(sample_ids.shape[0]), d, dtype)
            rel_p[sample_ids] = rel_p_s
            rel_v[sample_ids] = rel_v_s
            target_rel[sample_ids] = target_s
            flight_time[sample_ids] = flight_s
    else:
        rel_p, rel_v, target_rel, flight_time = _sample_toss_state(env, cfg, n, d, dtype)
        roll = torch.empty(n, device=d).uniform_(*cfg.get("roll", (-0.03, 0.03)))
        pitch = torch.empty(n, device=d).uniform_(*cfg.get("pitch", (-0.04, 0.04)))
        yaw = torch.empty(n, device=d).uniform_(*cfg.get("yaw", (-0.06, 0.06)))
        quat_w = _quat_from_euler_xyz(roll, pitch, yaw)
        ang_vel = torch.stack(
            [
                torch.empty(n, device=d).uniform_(*cfg.get("ang_vel_x", (-0.20, 0.20))),
                torch.empty(n, device=d).uniform_(*cfg.get("ang_vel_y", (-0.20, 0.20))),
                torch.empty(n, device=d).uniform_(*cfg.get("ang_vel_z", (-0.30, 0.30))),
            ],
            dim=-1,
        )

    pos_w = root_pos + quat_apply(root_yaw, rel_p)
    vel_w = robot.data.root_lin_vel_w[ids_throw] + quat_apply(root_yaw, rel_v)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=ids_throw)
    vel6 = torch.zeros((n, 6), device=d)
    vel6[:, 0:3] = vel_w
    vel6[:, 3:6] = ang_vel
    obj.write_root_velocity_to_sim(vel6, env_ids=ids_throw)

    env._urop_toss_done[ids_throw] += 1
    env._urop_toss_active[ids_throw] = True
    env._urop_toss_active_start_step[ids_throw] = env.episode_length_buf[ids_throw]
    env._urop_object_scene_visible[ids_throw] = True
    env._urop_pending_toss_valid[ids_throw] = False
    env._urop_obj_obs_cache_global_step = -1
    env._urop_obj_obs_cache_episode_len[ids_throw] = -1
    env._urop_last_toss_spawn_rel[ids_throw] = rel_p
    env._urop_last_toss_target_rel[ids_throw] = target_rel
    env._urop_last_toss_flight_time[ids_throw] = flight_time
