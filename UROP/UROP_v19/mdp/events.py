from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import get_controlled_joint_indices, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = scene_objects_cfg.OBJECT_BASE_SIZE

# Hidden task labels used only inside simulator events/rewards/critic. The actor never receives this task id.
TASK_HIDDEN = 0                  # no tag / object parked below the floor
TASK_VISIBLE_STATIC = 1          # tag visible, object held far away, never handed over
TASK_APPROACH_NO_RELEASE = 2     # tag visible and moving, but stops outside commit zone
TASK_HANDOVER_RELEASE = 3        # tag visible, slow handover, then released for whole-body receiving
TASK_NAMES = ("hidden", "visible_static", "approach_no_release", "handover_release")


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
    return _quat_from_yaw(yaw)


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1
    )


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


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smoothstep01_derivative(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return 6.0 * x * (1.0 - x)


def _ensure_urop_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype

    # Legacy names kept so v15 rewards/terminations/observations remain compatible.
    # In v19, "toss_active" means "receive/commit phase active", not "tag is visible".
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

    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((n, 2), device=d, dtype=dtype)
    if not hasattr(env, "_urop_spawn_root_pos"):
        env._urop_spawn_root_pos = torch.zeros((n, 3), device=d, dtype=dtype)
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

    if not hasattr(env, "_urop_handover_task"):
        env._urop_handover_task = torch.zeros(n, dtype=torch.long, device=d)
    if not hasattr(env, "_urop_handover_released"):
        env._urop_handover_released = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_handover_progress"):
        env._urop_handover_progress = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_start_s"):
        env._urop_handover_start_s = torch.full((n, 1), 999.0, device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_arrive_s"):
        env._urop_handover_arrive_s = torch.full((n, 1), 999.0, device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_release_s"):
        env._urop_handover_release_s = torch.full((n, 1), 999.0, device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_commit_lead_s"):
        env._urop_handover_commit_lead_s = torch.full((n, 1), 0.35, device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_start_rel"):
        env._urop_handover_start_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_goal_rel"):
        env._urop_handover_goal_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_stop_rel"):
        env._urop_handover_stop_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_static_rel"):
        env._urop_handover_static_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_handover_release_vel_rel"):
        env._urop_handover_release_vel_rel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_visible_truth"):
        env._urop_obj_visible_truth = torch.zeros((n, 1), dtype=torch.bool, device=d)

    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d, dtype=dtype).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), scene_objects_cfg.OBJECT_DEFAULT_MASS, device=d, dtype=dtype)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.8, device=d, dtype=dtype)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.02, device=d, dtype=dtype)
    if not hasattr(env, "_urop_mass_class_idx"):
        env._urop_mass_class_idx = torch.zeros((n, 1), dtype=torch.long, device=d)
    if not hasattr(env, "_urop_object_prior_one_hot"):
        prior = torch.zeros((n, 4), device=d, dtype=dtype)
        prior[:, 0] = 1.0
        env._urop_object_prior_one_hot = prior
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
    if not hasattr(env, "_urop_obj_obs_hold_prob"):
        env._urop_obj_obs_hold_prob = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos_noise_std"):
        env._urop_obj_obs_pos_noise_std = torch.full((n, 1), 0.01, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel_noise_std"):
        env._urop_obj_obs_vel_noise_std = torch.full((n, 1), 0.05, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos_bias"):
        env._urop_obj_obs_pos_bias = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos_scale"):
        env._urop_obj_obs_pos_scale = torch.ones((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel_scale"):
        env._urop_obj_obs_vel_scale = torch.ones((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_cache_global_step"):
        env._urop_obj_obs_cache_global_step = -1
    if not hasattr(env, "_urop_obj_obs_cache_episode_len"):
        env._urop_obj_obs_cache_episode_len = torch.full((n,), -1, device=d, dtype=torch.long)


def _apply_physx_mass_best_effort(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, mass: torch.Tensor) -> None:
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


def _sample_mass_classes(
    env: "ManagerBasedRLEnv",
    num_samples: int,
    mass_class_ranges: dict | None,
    mass_class_probabilities: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype
    mass_class_ranges = mass_class_ranges or {
        "light": (1.0, 2.5),
        "medium": (2.5, 4.5),
        "heavy": (4.5, 8.0),
    }
    probs_cfg = mass_class_probabilities or {"light": 0.34, "medium": 0.33, "heavy": 0.33}
    class_names = ("light", "medium", "heavy")
    probs = torch.tensor([float(probs_cfg.get(name, 0.0)) for name in class_names], device=d, dtype=dtype)
    if torch.sum(probs) <= 1.0e-6:
        probs[:] = 1.0
    probs = probs / torch.sum(probs)
    sampled_zero_based = torch.multinomial(probs, num_samples, replacement=True)
    mass = torch.empty((num_samples, 1), device=d, dtype=dtype)
    for i, name in enumerate(class_names):
        mask = sampled_zero_based == i
        if torch.any(mask):
            lo, hi = mass_class_ranges.get(name, mass_class_ranges[class_names[min(i, len(class_names) - 1)]])
            mass[mask, 0] = torch.empty(int(mask.sum()), device=d, dtype=dtype).uniform_(float(lo), float(hi))
    # Index 0 is unknown/no visible tag. Visible object classes are 1, 2, 3.
    return sampled_zero_based.to(torch.long) + 1, mass


def randomize_receive_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass_range=(1.0, 6.0),
    friction_range=(0.25, 1.50),
    restitution_range=(0.00, 0.18),
    size_scale_range=(0.80, 1.25),
    apply_physx: bool = True,
    mass_class_ranges: dict | None = None,
    mass_class_probabilities: dict | None = None,
    prior_unknown_prob: float = 0.03,
    prior_mismatch_prob: float = 0.04,
) -> None:
    """Randomize object physical properties and the tag-derived mass prior.

    v19 keeps the 104-dim actor contract by reusing the old 4-dim mode_one_hot as:
    [unknown/no visible tag, light, medium, heavy]. The true mass is continuous inside
    each class range; occasional unknown/mismatched priors prevent over-trusting the tag.
    """
    _ensure_urop_buffers(env)
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype
    n = int(env_ids.shape[0])

    if mass_class_ranges is not None or mass_class_probabilities is not None:
        class_idx, mass = _sample_mass_classes(env, n, mass_class_ranges, mass_class_probabilities)
    else:
        mass = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*mass_range)
        class_idx = torch.ones((n,), device=d, dtype=torch.long)
        class_idx[mass[:, 0] >= 2.5] = 2
        class_idx[mass[:, 0] >= 4.5] = 3

    friction = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*restitution_range)
    scale = torch.empty((n, 3), device=d, dtype=dtype).uniform_(*size_scale_range)
    size = torch.tensor(DEFAULT_BOX_SIZE, device=d, dtype=dtype).unsqueeze(0).repeat(n, 1) * scale

    # The spawned cuboid collision remains the nominal scene size. The size buffer is still useful
    # for critic/reward conditioning and debugging; true per-env collision resizing requires a
    # separate RigidObjectCollection/multi-asset refactor.
    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = size
    env._urop_mass_class_idx[env_ids, 0] = class_idx

    prior_idx = class_idx.clone()
    if prior_mismatch_prob > 0.0:
        mismatch = torch.rand(n, device=d) < float(prior_mismatch_prob)
        # Adjacent-ish corruption: light<->medium, heavy<->medium, with a little randomness.
        alt = torch.randint(1, 4, (n,), device=d)
        alt = torch.where(alt == prior_idx, 2 + (prior_idx == 2).long(), alt)
        alt = torch.clamp(alt, 1, 3)
        prior_idx = torch.where(mismatch, alt, prior_idx)

    prior = torch.zeros((n, 4), device=d, dtype=dtype)
    prior.scatter_(1, prior_idx.view(-1, 1), 1.0)
    if prior_unknown_prob > 0.0:
        unknown = torch.rand(n, device=d) < float(prior_unknown_prob)
        prior[unknown] = 0.0
        prior[unknown, 0] = 1.0
    env._urop_object_prior_one_hot[env_ids] = prior

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
    dtype = env.scene["robot"].data.root_pos_w.dtype
    n = int(env_ids.shape[0])

    robot_friction = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*friction_range)
    robot_restitution = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*restitution_range)
    floor_friction = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*floor_friction_range)
    # Ground is shared; fold floor variation into robot-side material per env.
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


def randomize_object_observation(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    projected_gravity_noise_std_range=(0.005, 0.030),
    base_ang_vel_noise_std_range=(0.01, 0.08),
    joint_pos_noise_std_range=(0.002, 0.020),
    joint_vel_noise_std_range=(0.02, 0.20),
    obj_pos_noise_range=(0.005, 0.060),
    obj_vel_noise_range=(0.03, 0.35),
    drop_prob_range=(0.00, 0.25),
    hold_prob_range=(0.00, 0.20),
    alpha_range=(0.20, 0.90),
    depth_scale_range=(0.90, 1.10),
    lateral_scale_range=(0.90, 1.10),
    height_scale_range=(0.92, 1.08),
    pos_bias_range=(-0.035, 0.035),
    vel_scale_range=(0.75, 1.25),
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype
    n = int(env_ids.shape[0])

    env._urop_projected_gravity_noise_std[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(
        *projected_gravity_noise_std_range
    )
    env._urop_base_ang_vel_noise_std[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(
        *base_ang_vel_noise_std_range
    )
    env._urop_joint_pos_noise_std[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*joint_pos_noise_std_range)
    env._urop_joint_vel_noise_std[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*joint_vel_noise_std_range)
    env._urop_obj_obs_pos_noise_std[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*obj_pos_noise_range)
    env._urop_obj_obs_vel_noise_std[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*obj_vel_noise_range)
    env._urop_obj_obs_drop_prob[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*drop_prob_range)
    env._urop_obj_obs_hold_prob[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*hold_prob_range)
    env._urop_obj_obs_alpha[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*alpha_range)

    pos_scale = torch.stack(
        [
            torch.empty(n, device=d, dtype=dtype).uniform_(*depth_scale_range),
            torch.empty(n, device=d, dtype=dtype).uniform_(*lateral_scale_range),
            torch.empty(n, device=d, dtype=dtype).uniform_(*height_scale_range),
        ],
        dim=-1,
    )
    env._urop_obj_obs_pos_scale[env_ids] = pos_scale
    env._urop_obj_obs_pos_bias[env_ids] = torch.empty((n, 3), device=d, dtype=dtype).uniform_(*pos_bias_range)
    env._urop_obj_obs_vel_scale[env_ids] = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*vel_scale_range)


def _sample_wait_times(env: "ManagerBasedRLEnv", num_envs: int, stage: int, wait_time_ranges: dict) -> torch.Tensor:
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype
    key = f"stage{stage}"
    if key not in wait_time_ranges:
        return torch.full((num_envs, 1), 999.0, device=d, dtype=dtype)
    low, high = wait_time_ranges[key]
    return torch.empty((num_envs, 1), device=d, dtype=dtype).uniform_(float(low), float(high))


def _sample_task_types(
    env: "ManagerBasedRLEnv",
    num_envs: int,
    stage: int,
    task_probability_by_stage: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = task_probability_by_stage or {}
    stage_cfg = cfg.get(f"stage{stage}", cfg.get("stage3", {}))
    default = {
        "hidden": 0.08,
        "visible_static": 0.22,
        "approach_no_release": 0.20,
        "handover_release": 0.50,
    }
    names = TASK_NAMES
    probs = torch.tensor(
        [float(stage_cfg.get(name, default[name])) for name in names],
        device=env.device,
        dtype=env.scene["robot"].data.root_pos_w.dtype,
    )
    if torch.sum(probs) <= 1.0e-6:
        probs[:] = 1.0
    probs = probs / torch.sum(probs)
    sampled = torch.multinomial(probs, num_envs, replacement=True).to(torch.long)
    handover_prob = probs[TASK_HANDOVER_RELEASE].view(1, 1).repeat(num_envs, 1)
    return sampled, handover_prob


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
    dtype = env.scene["robot"].data.root_pos_w.dtype
    pos_min = torch.empty((scene_objects_cfg.EXPECTED_ACTION_DIM,), device=d, dtype=dtype)
    pos_max = torch.empty((scene_objects_cfg.EXPECTED_ACTION_DIM,), device=d, dtype=dtype)

    for index, name in enumerate(scene_objects_cfg.CONTROLLED_JOINT_NAMES):
        if "wrist" in name:
            rng = pos_ranges["wrist"]
        elif "shoulder" in name or "elbow" in name:
            rng = pos_ranges["arm"]
        elif "waist" in name:
            rng = pos_ranges["waist"]
        else:
            rng = pos_ranges["lower_body"]
        pos_min[index] = float(rng[0])
        pos_max[index] = float(rng[1])

    pos_noise = torch.rand((n, scene_objects_cfg.EXPECTED_ACTION_DIM), device=d, dtype=dtype) * (pos_max - pos_min) + pos_min
    vel_noise = torch.empty((n, scene_objects_cfg.EXPECTED_ACTION_DIM), device=d, dtype=dtype).uniform_(*vel_range)
    return pos_noise, vel_noise


def _sample_uniform_triplet(cfg: dict, prefix: str, num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(
        [
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg[f"{prefix}_x"]),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg[f"{prefix}_y"]),
            torch.empty(num_samples, device=device, dtype=dtype).uniform_(*cfg[f"{prefix}_z"]),
        ],
        dim=-1,
    )


def _select_stage_cfg(stage: int, stage_cfgs: dict) -> dict:
    if stage <= 1:
        return stage_cfgs.get("stage1", stage_cfgs.get("stage3", {}))
    if stage == 2:
        return stage_cfgs.get("stage2", stage_cfgs.get("stage3", {}))
    return stage_cfgs.get("stage3", {})


def _sample_handover_trajectories(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage: int,
    handover_trajectory_by_stage: dict,
) -> None:
    _ensure_urop_buffers(env)
    cfg = _select_stage_cfg(stage, handover_trajectory_by_stage)
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype
    n = int(env_ids.shape[0])

    env._urop_handover_start_rel[env_ids] = _sample_uniform_triplet(cfg, "start", n, d, dtype)
    env._urop_handover_goal_rel[env_ids] = _sample_uniform_triplet(cfg, "goal", n, d, dtype)
    env._urop_handover_stop_rel[env_ids] = _sample_uniform_triplet(cfg, "stop", n, d, dtype)
    env._urop_handover_static_rel[env_ids] = _sample_uniform_triplet(cfg, "static", n, d, dtype)

    wait_s = _sample_wait_times(env, n, stage, cfg.get("start_wait_s", {f"stage{stage}": (0.3, 1.4)}))
    duration_low, duration_high = cfg.get("move_duration_s", (1.0, 2.8))
    duration_s = torch.empty((n, 1), device=d, dtype=dtype).uniform_(float(duration_low), float(duration_high))
    pause_low, pause_high = cfg.get("pre_release_pause_s", (0.10, 0.55))
    pause_s = torch.empty((n, 1), device=d, dtype=dtype).uniform_(float(pause_low), float(pause_high))
    commit_low, commit_high = cfg.get("commit_lead_s", (0.25, 0.55))
    commit_lead_s = torch.empty((n, 1), device=d, dtype=dtype).uniform_(float(commit_low), float(commit_high))

    env._urop_handover_start_s[env_ids] = wait_s
    env._urop_handover_arrive_s[env_ids] = wait_s + duration_s
    env._urop_handover_release_s[env_ids] = wait_s + duration_s + pause_s
    env._urop_handover_commit_lead_s[env_ids] = commit_lead_s

    env._urop_handover_release_vel_rel[env_ids] = torch.stack(
        [
            torch.empty(n, device=d, dtype=dtype).uniform_(*cfg.get("release_vel_x", (-0.10, 0.04))),
            torch.empty(n, device=d, dtype=dtype).uniform_(*cfg.get("release_vel_y", (-0.05, 0.05))),
            torch.empty(n, device=d, dtype=dtype).uniform_(*cfg.get("release_vel_z", (-0.10, 0.08))),
        ],
        dim=-1,
    )


def reset_handover_episode(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    park: dict,
    task_probability_by_stage: dict | None = None,
    handover_trajectory_by_stage: dict | None = None,
    joint_noise: dict | None = None,
    root_xy_range=(-0.02, 0.02),
    root_yaw_range=(-0.05, 0.05),
    object_randomization: dict | None = None,
    robot_material_randomization: dict | None = None,
    floor_material_randomization: dict | None = None,
    observation_randomization: dict | None = None,
) -> None:
    """Reset v19 episodes.

    The key change vs v15 is that object visibility and receiving obligation are sampled independently:
    many episodes contain a visible tagged box that is not handed over. This removes the shortcut
    "tag_visible=1 => immediately hug".
    """
    _ensure_urop_buffers(env)

    object_randomization = object_randomization or {}
    robot_material_randomization = robot_material_randomization or {}
    floor_material_randomization = floor_material_randomization or {}
    observation_randomization = observation_randomization or {}
    handover_trajectory_by_stage = handover_trajectory_by_stage or {}

    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    dtype = robot.data.root_pos_w.dtype
    n = int(env_ids.shape[0])

    stage = _get_stage(env)
    task_type, handover_probability = _sample_task_types(env, n, stage, task_probability_by_stage)
    should_receive = task_type == TASK_HANDOVER_RELEASE

    env._urop_toss_done[env_ids] = 0
    env._urop_toss_active[env_ids] = False
    env._urop_should_toss[env_ids] = should_receive
    env._urop_toss_wait_s[env_ids] = 999.0
    env._urop_toss_probability[env_ids] = handover_probability
    env._urop_last_toss_spawn_rel[env_ids] = 0.0
    env._urop_last_toss_target_rel[env_ids] = 0.0
    env._urop_last_toss_flight_time[env_ids] = 0.0

    env._urop_handover_task[env_ids] = task_type
    env._urop_handover_released[env_ids] = False
    env._urop_handover_progress[env_ids] = 0.0
    env._urop_obj_visible_truth[env_ids] = (task_type != TASK_HIDDEN).unsqueeze(-1)

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
    env._urop_obj_obs_cache_global_step = -1
    env._urop_obj_obs_cache_episode_len[env_ids] = -1

    env._urop_ready_joint_pos[env_ids] = robot.data.default_joint_pos[env_ids].clone()

    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    xy_noise = torch.empty((n, 2), device=d, dtype=dtype).uniform_(*root_xy_range)
    yaw_noise = torch.empty((n,), device=d, dtype=dtype).uniform_(*root_yaw_range)
    yaw_quat = _quat_from_euler_xyz(torch.zeros(n, device=d), torch.zeros(n, device=d), yaw_noise)

    root_pos = default_root_state[:, 0:3]
    root_pos[:, 0:2] += xy_noise
    root_quat = quat_mul(default_root_state[:, 3:7], yaw_quat)
    root_vel = torch.zeros((n, 6), device=d, dtype=dtype)

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
    env._urop_spawn_root_pos[env_ids] = root_pos
    env._urop_spawn_yaw[env_ids, 0] = torch.atan2(
        2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
        1.0 - 2.0 * (root_quat[:, 2] ** 2 + root_quat[:, 3] ** 2),
    )

    env._urop_toss_episode_count += should_receive.to(dtype=torch.int64).sum()
    env._urop_no_toss_episode_count += (~should_receive).to(dtype=torch.int64).sum()

    # Stage-conditioned mass prior distribution: object mass is now part of the task, not a hidden nuisance only.
    obj_rand = dict(object_randomization)
    mass_probs_by_stage = obj_rand.pop("mass_class_prob_by_stage", None)
    if mass_probs_by_stage is not None:
        obj_rand["mass_class_probabilities"] = mass_probs_by_stage.get(f"stage{stage}", mass_probs_by_stage.get("stage3", None))
    randomize_receive_object(env, env_ids, **obj_rand)
    randomize_robot_contact_material(
        env,
        env_ids,
        friction_range=robot_material_randomization.get("friction_range", (0.70, 1.00)),
        restitution_range=robot_material_randomization.get("restitution_range", (0.00, 0.02)),
        floor_friction_range=floor_material_randomization.get("friction_range", (0.75, 1.05)),
        apply_physx=robot_material_randomization.get("apply_physx", True),
    )
    randomize_object_observation(env, env_ids, **observation_randomization)

    _sample_handover_trajectories(env, env_ids, stage, handover_trajectory_by_stage)

    park_rel = torch.stack(
        [
            torch.empty(n, device=d, dtype=dtype).uniform_(*park["pos_x"]),
            torch.empty(n, device=d, dtype=dtype).uniform_(*park["pos_y"]),
            torch.empty(n, device=d, dtype=dtype).uniform_(*park["pos_z"]),
        ],
        dim=-1,
    )

    rel_initial = env._urop_handover_start_rel[env_ids].clone()
    rel_initial = torch.where((task_type == TASK_VISIBLE_STATIC).unsqueeze(-1), env._urop_handover_static_rel[env_ids], rel_initial)
    rel_initial = torch.where((task_type == TASK_HIDDEN).unsqueeze(-1), park_rel, rel_initial)

    pos_w = root_pos + quat_apply(_yaw_quat(root_quat), rel_initial)
    roll = torch.empty(n, device=d, dtype=dtype).uniform_(-0.04, 0.04)
    pitch = torch.empty(n, device=d, dtype=dtype).uniform_(-0.05, 0.05)
    yaw = torch.empty(n, device=d, dtype=dtype).uniform_(-0.20, 0.20)
    quat_w = _quat_from_euler_xyz(roll, pitch, yaw)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=d, dtype=dtype), env_ids=env_ids)


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
            [robot.data.root_lin_vel_w[env_ids].clone(), robot.data.root_ang_vel_w[env_ids].clone()], dim=-1
        )

    d = env.device
    dtype = root_vel.dtype
    n = int(env_ids.shape[0])
    dv_xy = torch.empty((n, 2), device=d, dtype=dtype).uniform_(*xy_range)
    dv_z = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*z_velocity_range)
    dyaw = torch.empty((n, 1), device=d, dtype=dtype).uniform_(*yaw_range)

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


def _held_trajectory_state(
    env: "ManagerBasedRLEnv",
    ids: torch.Tensor,
    use_stop_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = env.scene["robot"].data.root_pos_w.dtype
    t = env.episode_length_buf[ids].float().unsqueeze(-1).to(dtype=dtype) * float(env.step_dt)
    start_s = env._urop_handover_start_s[ids]
    arrive_s = env._urop_handover_arrive_s[ids]
    duration = torch.clamp(arrive_s - start_s, min=float(env.step_dt))
    raw = (t - start_s) / duration
    progress = _smoothstep01(raw)
    dpdt = _smoothstep01_derivative(raw) / duration

    start_rel = env._urop_handover_start_rel[ids]
    goal_rel = torch.where(use_stop_target.unsqueeze(-1), env._urop_handover_stop_rel[ids], env._urop_handover_goal_rel[ids])
    rel = start_rel * (1.0 - progress) + goal_rel * progress
    rel_vel = (goal_rel - start_rel) * dpdt
    return rel, rel_vel, progress


def handover_object_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    commit_x: float = 0.58,
    debug_print: bool = False,
    debug_print_rate_s: float = 0.75,
) -> None:
    """Drive visible object episodes and release only true handover episodes.

    This event runs every policy step. Before release, the object is continuously rewritten to
    simulate a human-held box. Static and no-release episodes keep the tag visible but never set
    the receive-active gate, so catching rewards/termination do not fire.
    """
    _ensure_urop_buffers(env)

    if env_ids.numel() == 0:
        return

    obj = env.scene["object"]
    d = env.device
    dtype = obj.data.root_pos_w.dtype
    ids = env_ids
    task = env._urop_handover_task[ids]
    t = env.episode_length_buf[ids].float().unsqueeze(-1).to(dtype=dtype) * float(env.step_dt)

    visible = task != TASK_HIDDEN
    handover = task == TASK_HANDOVER_RELEASE
    approach_no_release = task == TASK_APPROACH_NO_RELEASE
    static_visible = task == TASK_VISIBLE_STATIC

    # Visibility truth is independent of whether the robot should receive the object.
    env._urop_obj_visible_truth[ids] = visible.unsqueeze(-1)

    release_now = handover & (~env._urop_handover_released[ids]) & (t[:, 0] >= env._urop_handover_release_s[ids, 0])
    if torch.any(release_now):
        rel_release = env._urop_handover_goal_rel[ids[release_now]]
        yaw_q_release = _quat_from_yaw(env._urop_spawn_yaw[ids[release_now], 0])
        pos_release = env._urop_spawn_root_pos[ids[release_now]] + quat_apply(yaw_q_release, rel_release)
        quat_release = obj.data.root_quat_w[ids[release_now]]
        obj.write_root_pose_to_sim(torch.cat([pos_release, quat_release], dim=-1), env_ids=ids[release_now])
        release_vel_w = quat_apply(yaw_q_release, env._urop_handover_release_vel_rel[ids[release_now]])
        vel6 = torch.zeros((int(release_now.sum()), 6), device=d, dtype=dtype)
        vel6[:, 0:3] = release_vel_w
        vel6[:, 3:6] = 0.0
        obj.write_root_velocity_to_sim(vel6, env_ids=ids[release_now])
        env._urop_handover_released[ids[release_now]] = True
        env._urop_toss_done[ids[release_now]] = 1

    not_released = ~(handover & env._urop_handover_released[ids])
    write_mask = visible & not_released
    if torch.any(write_mask):
        ids_write = ids[write_mask]
        task_write = env._urop_handover_task[ids_write]
        use_stop_target = task_write == TASK_APPROACH_NO_RELEASE
        rel, rel_vel, progress = _held_trajectory_state(env, ids_write, use_stop_target=use_stop_target)
        rel = torch.where((task_write == TASK_VISIBLE_STATIC).unsqueeze(-1), env._urop_handover_static_rel[ids_write], rel)
        rel_vel = torch.where((task_write == TASK_VISIBLE_STATIC).unsqueeze(-1), torch.zeros_like(rel_vel), rel_vel)
        env._urop_handover_progress[ids_write] = torch.where(
            (task_write == TASK_VISIBLE_STATIC).unsqueeze(-1), torch.zeros_like(progress), progress
        )

        yaw_q = _quat_from_yaw(env._urop_spawn_yaw[ids_write, 0])
        pos_w = env._urop_spawn_root_pos[ids_write] + quat_apply(yaw_q, rel)
        # Keep orientation simple but not perfectly fixed: small randomized reset orientation is preserved.
        quat_w = obj.data.root_quat_w[ids_write]
        vel_w = quat_apply(yaw_q, rel_vel)
        vel6 = torch.zeros((ids_write.shape[0], 6), device=d, dtype=dtype)
        vel6[:, 0:3] = vel_w
        obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=ids_write)
        obj.write_root_velocity_to_sim(vel6, env_ids=ids_write)

    # Hidden objects remain parked below the floor so they cannot accidentally contact the robot.
    hidden_mask = task == TASK_HIDDEN
    if torch.any(hidden_mask):
        ids_hidden = ids[hidden_mask]
        rel = torch.stack(
            [
                torch.full((ids_hidden.shape[0],), 1.65, device=d, dtype=dtype),
                torch.zeros((ids_hidden.shape[0],), device=d, dtype=dtype),
                torch.full((ids_hidden.shape[0],), -0.58, device=d, dtype=dtype),
            ],
            dim=-1,
        )
        yaw_q = _quat_from_yaw(env._urop_spawn_yaw[ids_hidden, 0])
        pos_w = env._urop_spawn_root_pos[ids_hidden] + quat_apply(yaw_q, rel)
        quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d, dtype=dtype).repeat(ids_hidden.shape[0], 1)
        obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=ids_hidden)
        obj.write_root_velocity_to_sim(torch.zeros((ids_hidden.shape[0], 6), device=d, dtype=dtype), env_ids=ids_hidden)

    # Commit/receive-active gate: only true handover episodes become active, and only near release.
    commit_by_time = handover & (t[:, 0] >= (env._urop_handover_release_s[ids, 0] - env._urop_handover_commit_lead_s[ids, 0]))
    commit_by_position = handover & (obj.data.root_pos_w[ids, 0] - env._urop_spawn_root_pos[ids, 0] < float(commit_x))
    active = commit_by_time | commit_by_position | (handover & env._urop_handover_released[ids])
    env._urop_toss_active[ids] = active

    # No-receive episodes must never trigger drop termination or catch rewards.
    env._urop_toss_active[ids[static_visible | approach_no_release]] = False
    env._urop_toss_active[ids[hidden_mask]] = False

    env._urop_obj_obs_cache_global_step = -1
    env._urop_obj_obs_cache_episode_len[ids] = -1

    if debug_print:
        if not hasattr(env, "_urop_v19_last_debug_print_step"):
            env._urop_v19_last_debug_print_step = -10**9
        rate_steps = max(int(float(debug_print_rate_s) / max(float(env.step_dt), 1.0e-6)), 1)
        if int(env.common_step_counter) - int(env._urop_v19_last_debug_print_step) >= rate_steps:
            env._urop_v19_last_debug_print_step = int(env.common_step_counter)
            all_tasks = env._urop_handover_task.detach()
            task_counts = torch.bincount(all_tasks.clamp(0, 3), minlength=4).detach().cpu().tolist()
            class_counts = torch.bincount(env._urop_mass_class_idx[:, 0].clamp(0, 3), minlength=4).detach().cpu().tolist()
            visible_count = int(env._urop_obj_visible_truth.sum().detach().cpu())
            active_count = int(env._urop_toss_active.sum().detach().cpu())
            released_count = int(env._urop_handover_released.sum().detach().cpu())
            mean_mass = float(env._urop_box_mass.mean().detach().cpu())
            print(
                "[UROP_v19 debug] "
                f"stage={_get_stage(env)} step={int(env.common_step_counter)} "
                f"tasks hidden/static/approach/handover={task_counts} "
                f"mass classes unk/light/med/heavy={class_counts} mean_mass={mean_mass:.2f}kg "
                f"visible_truth={visible_count}/{env.num_envs} active_receive={active_count}/{env.num_envs} "
                f"released={released_count}/{env.num_envs}"
            )
