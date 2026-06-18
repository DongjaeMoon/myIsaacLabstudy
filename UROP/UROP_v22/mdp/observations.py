from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterable

import torch

from .. import scene_objects_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# UROP-v22 design constants.
# -----------------------------------------------------------------------------
# Camera convention used for policy object observations:
#   - "opencv": x right, y down, z forward. This matches the usual AprilTag/OpenCV
#     tvec convention and is therefore the default actor-policy contract.
#   - "body": x forward, y left, z up in the selected camera/head body frame.
CAMERA_CANDIDATE_BODY_NAMES = (
    "head_camera_link",
    "head_link",
    "torso_link",
    "pelvis",
    "base_link",
)
DEFAULT_CAMERA_OFFSET_B = (0.18, 0.0, 0.34)
HOLD_ANCHOR_B = (0.30, 0.0, 0.23)
GRAVITY_W = (0.0, 0.0, -1.0)

# Episode types. They are privileged labels and must not be fed to the actor.
EP_TOSS = 0
EP_DELAYED_TOSS = 1
EP_NO_TOSS = 2
EP_PASS_BY = 3
EP_NEAR_MISS = 4
NUM_EPISODE_TYPES = 5


# -----------------------------------------------------------------------------
# Generic helpers.
# -----------------------------------------------------------------------------
def _num_envs(env: "ManagerBasedRLEnv") -> int:
    return int(getattr(env, "num_envs", getattr(env.scene, "num_envs")))


def _device(env: "ManagerBasedRLEnv") -> torch.device:
    dev = getattr(env, "device", None)
    if dev is not None:
        return torch.device(dev)
    try:
        return torch.device(env.scene["robot"].device)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_ids(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(_num_envs(env), device=_device(env), dtype=torch.long)
    return env_ids.to(device=_device(env), dtype=torch.long)


def _as_tensor(values: Iterable[float] | torch.Tensor, env: "ManagerBasedRLEnv") -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=_device(env), dtype=torch.float32)
    return torch.tensor(tuple(values), device=_device(env), dtype=torch.float32)


def _rand_uniform(shape: tuple[int, ...], low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high)


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # Isaac Lab stores quaternions as (w, x, y, z).
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(_quat_conjugate(q), v)


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _exp_reward(error: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(float(sigma), 1e-6)
    return torch.exp(-(error * error) / (sigma * sigma))


# -----------------------------------------------------------------------------
# Environment/task state cache.
# -----------------------------------------------------------------------------
def get_task_state(env: "ManagerBasedRLEnv") -> SimpleNamespace:
    """Return the per-env UROP-v22 task state cache.

    The state object stores only training/simulation bookkeeping. Actor observations
    still expose only the explicitly configured real-robot-available terms.
    """
    n = _num_envs(env)
    device = _device(env)
    state = getattr(env, "_urop_v22_state", None)
    if state is not None and getattr(state, "num_envs", None) == n and getattr(state, "device", None) == device:
        return state

    state = SimpleNamespace()
    state.num_envs = n
    state.device = device
    state.curriculum_stage = torch.zeros(n, device=device, dtype=torch.long)
    state.episode_type = torch.zeros(n, device=device, dtype=torch.long)
    state.release_step = torch.zeros(n, device=device, dtype=torch.long)
    state.has_released = torch.zeros(n, device=device, dtype=torch.bool)
    state.ever_released = torch.zeros(n, device=device, dtype=torch.bool)
    state.release_velocity_w = torch.zeros(n, 3, device=device)
    state.release_ang_velocity_w = torch.zeros(n, 3, device=device)
    state.held_position_w = torch.zeros(n, 3, device=device)
    state.held_quat_w = torch.zeros(n, 4, device=device)
    state.held_quat_w[:, 0] = 1.0
    state.target_anchor_b = _as_tensor(HOLD_ANCHOR_B, env).repeat(n, 1)
    state.tag_available = torch.ones(n, device=device, dtype=torch.bool)
    state.obs_tag_visible = torch.zeros(n, 1, device=device)
    state._obs_cache_step = -1
    state.hold_counter = torch.zeros(n, device=device, dtype=torch.long)
    state.drop_counter = torch.zeros(n, device=device, dtype=torch.long)
    state.hold_counter_step = -1
    state.object_size = _as_tensor(scene_objects_cfg.OBJECT_BASE_SIZE, env).repeat(n, 1)
    state.object_mass = torch.full((n, 1), float(scene_objects_cfg.OBJECT_DEFAULT_MASS), device=device)
    state.object_friction = torch.full((n, 1), 0.80, device=device)
    state.object_restitution = torch.full((n, 1), 0.02, device=device)
    state.obs_noise_scale = torch.ones(n, 1, device=device)
    state.push_scale = torch.ones(n, 1, device=device)
    state.last_action = torch.zeros(n, scene_objects_cfg.EXPECTED_ACTION_DIM, device=device)
    state.prev_action = torch.zeros_like(state.last_action)
    state.prev_prev_action = torch.zeros_like(state.last_action)
    state.action_initialized = torch.zeros(n, device=device, dtype=torch.bool)
    state.action_history_step = -1
    state.joint_id_cache = {}
    state.body_id_cache = {}
    state.last_hand_dist = torch.zeros(n, 2, device=device)
    setattr(env, "_urop_v22_state", state)
    return state


# -----------------------------------------------------------------------------
# Scene accessors.
# -----------------------------------------------------------------------------
def _get_robot(env: "ManagerBasedRLEnv", asset_name: str = "robot"):
    return env.scene[asset_name]


def _get_object(env: "ManagerBasedRLEnv", asset_name: str = "object"):
    return env.scene[asset_name]


def _get_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        return torch.zeros(_num_envs(env), 3, device=_device(env))
    return origins.to(_device(env))


def _root_pos_quat(asset) -> tuple[torch.Tensor, torch.Tensor]:
    data = asset.data
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        return data.root_pos_w, data.root_quat_w
    root = data.root_state_w
    return root[:, 0:3], root[:, 3:7]


def _root_lin_vel_w(asset) -> torch.Tensor:
    data = asset.data
    if hasattr(data, "root_lin_vel_w"):
        return data.root_lin_vel_w
    if hasattr(data, "root_state_w"):
        return data.root_state_w[:, 7:10]
    return torch.zeros(asset.num_instances, 3, device=asset.device)


def _root_ang_vel_w(asset) -> torch.Tensor:
    data = asset.data
    if hasattr(data, "root_ang_vel_w"):
        return data.root_ang_vel_w
    if hasattr(data, "root_state_w"):
        return data.root_state_w[:, 10:13]
    return torch.zeros(asset.num_instances, 3, device=asset.device)


def _root_lin_vel_b(asset) -> torch.Tensor:
    data = asset.data
    if hasattr(data, "root_lin_vel_b"):
        return data.root_lin_vel_b
    root_pos, root_quat = _root_pos_quat(asset)
    del root_pos
    return _quat_rotate_inverse(root_quat, _root_lin_vel_w(asset))


def _root_ang_vel_b(asset) -> torch.Tensor:
    data = asset.data
    if hasattr(data, "root_ang_vel_b"):
        return data.root_ang_vel_b
    root_pos, root_quat = _root_pos_quat(asset)
    del root_pos
    return _quat_rotate_inverse(root_quat, _root_ang_vel_w(asset))


def _controlled_joint_ids(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    key = "controlled"
    cached = state.joint_id_cache.get(key)
    if cached is not None:
        return cached

    robot = _get_robot(env)
    names = list(scene_objects_cfg.CONTROLLED_JOINT_NAMES)
    try:
        ids, resolved_names = robot.find_joints(names, preserve_order=True)
        del resolved_names
        ids = torch.as_tensor(ids, device=_device(env), dtype=torch.long)
    except Exception:
        all_names = list(getattr(robot, "joint_names", []))
        name_to_id = {name: i for i, name in enumerate(all_names)}
        ids = torch.tensor([name_to_id[name] for name in names], device=_device(env), dtype=torch.long)
    if ids.numel() != scene_objects_cfg.EXPECTED_ACTION_DIM:
        raise RuntimeError(f"Expected 29 controlled joints, got {ids.numel()}.")
    state.joint_id_cache[key] = ids
    return ids


def _controlled_pose_tensor(env: "ManagerBasedRLEnv", pose: dict[str, float]) -> torch.Tensor:
    values = [float(pose[name]) for name in scene_objects_cfg.CONTROLLED_JOINT_NAMES]
    return torch.tensor(values, device=_device(env), dtype=torch.float32)


def _find_body_id(env: "ManagerBasedRLEnv", body_names: tuple[str, ...]) -> int | None:
    state = get_task_state(env)
    key = tuple(body_names)
    if key in state.body_id_cache:
        return state.body_id_cache[key]

    robot = _get_robot(env)
    body_id: int | None = None
    for name in body_names:
        try:
            ids, resolved = robot.find_bodies(name, preserve_order=True)
            del resolved
            if len(ids) > 0:
                body_id = int(ids[0])
                break
        except Exception:
            pass
        all_names = list(getattr(robot, "body_names", []))
        if name in all_names:
            body_id = int(all_names.index(name))
            break
    state.body_id_cache[key] = body_id
    return body_id


def _body_pos_quat_vel_w(
    env: "ManagerBasedRLEnv",
    body_names: tuple[str, ...],
    fallback_offset_b: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = _get_robot(env)
    root_pos, root_quat = _root_pos_quat(robot)
    body_id = _find_body_id(env, body_names)
    data = robot.data
    if body_id is not None and hasattr(data, "body_pos_w") and hasattr(data, "body_quat_w"):
        pos = data.body_pos_w[:, body_id]
        quat = data.body_quat_w[:, body_id]
        if hasattr(data, "body_lin_vel_w"):
            lin_vel = data.body_lin_vel_w[:, body_id]
        else:
            lin_vel = _root_lin_vel_w(robot)
        return pos, quat, lin_vel

    if fallback_offset_b is None:
        fallback_offset_b = (0.0, 0.0, 0.0)
    offset = _as_tensor(fallback_offset_b, env).repeat(_num_envs(env), 1)
    pos = root_pos + _quat_rotate(root_quat, offset)
    return pos, root_quat, _root_lin_vel_w(robot)


def _camera_pose_vel_w(
    env: "ManagerBasedRLEnv",
    body_names: tuple[str, ...] = CAMERA_CANDIDATE_BODY_NAMES,
    fallback_offset_b: tuple[float, float, float] = DEFAULT_CAMERA_OFFSET_B,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _body_pos_quat_vel_w(env, body_names, fallback_offset_b)


def _body_to_camera_convention(v_body: torch.Tensor, camera_frame: str = "opencv") -> torch.Tensor:
    if camera_frame in ("body", "robot", "x-forward"):
        return v_body
    if camera_frame in ("opencv", "optical", "camera"):
        # Body/head convention: x forward, y left, z up.
        # OpenCV optical convention: x right, y down, z forward.
        return torch.stack((-v_body[..., 1], -v_body[..., 2], v_body[..., 0]), dim=-1)
    raise ValueError(f"Unsupported camera_frame='{camera_frame}'. Use 'opencv' or 'body'.")


def _object_pos_quat_vel_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = _get_object(env)
    obj_pos, obj_quat = _root_pos_quat(obj)
    return obj_pos, obj_quat, _root_lin_vel_w(obj), _root_ang_vel_w(obj)


def _object_rel_in_root_frame(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = _get_robot(env)
    root_pos, root_quat = _root_pos_quat(robot)
    obj_pos, _obj_quat, obj_lin_vel, obj_ang_vel = _object_pos_quat_vel_w(env)
    rel_pos_b = _quat_rotate_inverse(root_quat, obj_pos - root_pos)
    rel_lin_vel_b = _quat_rotate_inverse(root_quat, obj_lin_vel - _root_lin_vel_w(robot))
    obj_ang_vel_b = _quat_rotate_inverse(root_quat, obj_ang_vel)
    return rel_pos_b, rel_lin_vel_b, obj_ang_vel_b


def _hand_positions_root_frame(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = _get_robot(env)
    root_pos, root_quat = _root_pos_quat(robot)
    left_pos_w, _left_q, _left_v = _body_pos_quat_vel_w(env, ("left_hand_palm_link", "left_wrist_yaw_link"))
    right_pos_w, _right_q, _right_v = _body_pos_quat_vel_w(env, ("right_hand_palm_link", "right_wrist_yaw_link"))
    left_b = _quat_rotate_inverse(root_quat, left_pos_w - root_pos)
    right_b = _quat_rotate_inverse(root_quat, right_pos_w - root_pos)
    return left_b, right_b


def _noise_like(x: torch.Tensor, base_std: float, env: "ManagerBasedRLEnv") -> torch.Tensor:
    if base_std <= 0.0:
        return torch.zeros_like(x)
    state = get_task_state(env)
    scale = state.obs_noise_scale
    while scale.dim() < x.dim():
        scale = scale.unsqueeze(-1)
    return torch.randn_like(x) * float(base_std) * scale


def _current_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    n = _num_envs(env)
    device = _device(env)
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        for attr in ("action", "prev_action", "_action", "_prev_action"):
            value = getattr(action_manager, attr, None)
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=torch.float32)
        try:
            term = action_manager.get_term("policy")
            for attr in ("raw_actions", "processed_actions", "action"):
                value = getattr(term, attr, None)
                if isinstance(value, torch.Tensor):
                    return value.to(device=device, dtype=torch.float32)
        except Exception:
            pass
    return torch.zeros(n, scene_objects_cfg.EXPECTED_ACTION_DIM, device=device)


def update_action_history(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update and return (current, previous, previous_previous) action buffers.

    Manager terms can call this from multiple rewards in the same step. The
    internal step guard prevents the history from being shifted more than once.
    """
    state = get_task_state(env)
    step = _global_step(env)
    if state.action_history_step == step:
        return state.last_action, state.prev_action, state.prev_prev_action

    current = _current_action(env)
    if current.shape[-1] != scene_objects_cfg.EXPECTED_ACTION_DIM:
        current = current[..., : scene_objects_cfg.EXPECTED_ACTION_DIM]
    state.prev_prev_action = state.prev_action.clone()
    state.prev_action = torch.where(state.action_initialized[:, None], state.last_action, current)
    state.last_action = current
    state.action_initialized[:] = True
    state.action_history_step = step
    return state.last_action, state.prev_action, state.prev_prev_action


# -----------------------------------------------------------------------------
# Visibility model and policy observations.
# -----------------------------------------------------------------------------
def _episode_step(env: "ManagerBasedRLEnv") -> torch.Tensor:
    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        return torch.zeros(_num_envs(env), device=_device(env), dtype=torch.long)
    return step_buf.to(device=_device(env), dtype=torch.long)


def _global_step(env: "ManagerBasedRLEnv") -> int:
    step = getattr(env, "common_step_counter", 0)
    try:
        return int(step)
    except Exception:
        return int(step.item())


def _compute_tag_visibility(
    env: "ManagerBasedRLEnv",
    max_range: float = 2.40,
    min_forward: float = 0.05,
    h_fov_rad: float = 1.50,
    v_fov_rad: float = 1.20,
    dropout_prob: float = 0.035,
) -> torch.Tensor:
    state = get_task_state(env)
    current_step = _global_step(env)
    if state._obs_cache_step == current_step:
        return state.obs_tag_visible

    cam_pos, cam_quat, _cam_vel = _camera_pose_vel_w(env)
    obj_pos, _obj_quat, _obj_lin_vel, _obj_ang_vel = _object_pos_quat_vel_w(env)
    rel_body = _quat_rotate_inverse(cam_quat, obj_pos - cam_pos)
    forward = rel_body[:, 0]
    lateral = rel_body[:, 1]
    vertical = rel_body[:, 2]
    dist = _safe_norm(rel_body)
    h_angle = torch.atan2(torch.abs(lateral), torch.clamp(forward, min=1e-4))
    v_angle = torch.atan2(torch.abs(vertical), torch.clamp(forward, min=1e-4))
    visible = (
        state.tag_available
        & (forward > min_forward)
        & (dist < max_range)
        & (h_angle < h_fov_rad * 0.5)
        & (v_angle < v_fov_rad * 0.5)
    )
    # Real AprilTag detections can intermittently drop even when geometrically visible.
    p = torch.clamp(float(dropout_prob) * state.obs_noise_scale.squeeze(-1), 0.0, 0.60)
    keep = torch.rand_like(p) > p
    visible = visible & keep
    state.obs_tag_visible = visible.to(torch.float32).unsqueeze(-1)
    state._obs_cache_step = current_step
    return state.obs_tag_visible


def projected_gravity(env: "ManagerBasedRLEnv", asset_name: str = "robot", noise_std: float = 0.010) -> torch.Tensor:
    robot = _get_robot(env, asset_name)
    data = robot.data
    if hasattr(data, "projected_gravity_b"):
        g = data.projected_gravity_b
    else:
        _root_pos, root_quat = _root_pos_quat(robot)
        gravity = _as_tensor(GRAVITY_W, env).repeat(_num_envs(env), 1)
        g = _quat_rotate_inverse(root_quat, gravity)
    return torch.clamp(g + _noise_like(g, noise_std, env), -1.5, 1.5)


def base_angular_velocity(env: "ManagerBasedRLEnv", asset_name: str = "robot", noise_std: float = 0.025) -> torch.Tensor:
    w_b = _root_ang_vel_b(_get_robot(env, asset_name))
    return torch.clamp(w_b + _noise_like(w_b, noise_std, env), -12.0, 12.0)


def controlled_joint_pos_rel(env: "ManagerBasedRLEnv", asset_name: str = "robot", noise_std: float = 0.006) -> torch.Tensor:
    robot = _get_robot(env, asset_name)
    ids = _controlled_joint_ids(env)
    q = robot.data.joint_pos[:, ids]
    q_ref = _controlled_pose_tensor(env, scene_objects_cfg.READY_POSE)
    out = q - q_ref
    return torch.clamp(out + _noise_like(out, noise_std, env), -3.5, 3.5)


def controlled_joint_velocities(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    scale: float = 1.0,
    noise_std: float = 0.20,
) -> torch.Tensor:
    robot = _get_robot(env, asset_name)
    ids = _controlled_joint_ids(env)
    qd = robot.data.joint_vel[:, ids]
    qd_noisy = qd + _noise_like(qd, noise_std, env)
    return torch.clamp(qd_noisy * float(scale), -8.0, 8.0)


def prev_actions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Do not add observation noise to previous action; the deployed controller knows
    # the action it sent. This is still a real-robot-available signal.
    return _current_action(env)


def object_rel_pos(
    env: "ManagerBasedRLEnv",
    camera_frame: str = "opencv",
    noise_std: float = 0.018,
    zero_when_not_visible: bool = True,
) -> torch.Tensor:
    cam_pos, cam_quat, _cam_lin_vel = _camera_pose_vel_w(env)
    obj_pos, _obj_quat, _obj_lin_vel, _obj_ang_vel = _object_pos_quat_vel_w(env)
    rel_body = _quat_rotate_inverse(cam_quat, obj_pos - cam_pos)
    rel = _body_to_camera_convention(rel_body, camera_frame)
    visible = _compute_tag_visibility(env)
    rel = rel + _noise_like(rel, noise_std, env)
    if zero_when_not_visible:
        rel = rel * visible
    return torch.clamp(rel, -4.0, 4.0)


def object_rel_lin_vel(
    env: "ManagerBasedRLEnv",
    camera_frame: str = "opencv",
    noise_std: float = 0.08,
    zero_when_not_visible: bool = True,
) -> torch.Tensor:
    cam_pos, cam_quat, cam_lin_vel = _camera_pose_vel_w(env)
    del cam_pos
    obj_pos, _obj_quat, obj_lin_vel, _obj_ang_vel = _object_pos_quat_vel_w(env)
    del obj_pos
    rel_body = _quat_rotate_inverse(cam_quat, obj_lin_vel - cam_lin_vel)
    rel = _body_to_camera_convention(rel_body, camera_frame)
    visible = _compute_tag_visibility(env)
    rel = rel + _noise_like(rel, noise_std, env)
    if zero_when_not_visible:
        rel = rel * visible
    return torch.clamp(rel, -8.0, 8.0)


def tag_visible(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _compute_tag_visibility(env)


# -----------------------------------------------------------------------------
# Privileged critic observations.
# -----------------------------------------------------------------------------
def mode_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    return torch.nn.functional.one_hot(state.episode_type.clamp(0, NUM_EPISODE_TYPES - 1), NUM_EPISODE_TYPES).to(torch.float32)


def toss_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    steps = _episode_step(env).to(torch.float32)
    release = state.release_step.to(torch.float32)
    dt = float(getattr(env, "step_dt", 0.02))
    time_to_release = torch.clamp((release - steps) * dt, -2.0, 2.0).unsqueeze(-1)
    time_since_release = torch.clamp((steps - release) * dt, -2.0, 4.0).unsqueeze(-1)
    released = state.has_released.to(torch.float32).unsqueeze(-1)
    one_hot = mode_one_hot(env)
    return torch.cat((one_hot, released, time_to_release, time_since_release), dim=-1)


def hold_condition(env: "ManagerBasedRLEnv", pos_tol: tuple[float, float, float] = (0.145, 0.175, 0.155)) -> torch.Tensor:
    """Strict demo-quality hold: chest pocket + slow object + two-hand bracket.

    Contact/force is intentionally not used here. The actor still sees only
    proprioception and AprilTag-like object pose/velocity.
    """
    state = get_task_state(env)
    rel_pos_b, rel_lin_vel_b, obj_ang_vel_b = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    tol = _as_tensor(pos_tol, env).view(1, 3)
    in_box = torch.all(torch.abs(err) < tol, dim=-1)
    slow = (_safe_norm(rel_lin_vel_b) < 0.30) & (_safe_norm(obj_ang_vel_b) < 1.35)
    left_err, right_err, lateral_order, depth_order = hand_side_errors(env)
    hand_wrap = (left_err < 0.26) & (right_err < 0.26) & (lateral_order > 0.5) & (depth_order > 0.5)
    valid_episode = (state.episode_type == EP_TOSS) | (state.episode_type == EP_DELAYED_TOSS)
    return (in_box & slow & hand_wrap & valid_episode & state.has_released).to(torch.float32).unsqueeze(-1)


def hold_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return hold_condition(env)


def drop_state(env: "ManagerBasedRLEnv", drop_z: float = 0.30) -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, _obj_q, _obj_v, _obj_w = _object_pos_quat_vel_w(env)
    dropped = (obj_pos[:, 2] < drop_z) & state.has_released & (state.episode_type != EP_NO_TOSS)
    return dropped.to(torch.float32).unsqueeze(-1)


def critic_robot_state(env: "ManagerBasedRLEnv", asset_name: str = "robot", torque_scale: float = 1.0) -> torch.Tensor:
    robot = _get_robot(env, asset_name)
    ids = _controlled_joint_ids(env)
    q_ref = _controlled_pose_tensor(env, scene_objects_cfg.READY_POSE)
    q = robot.data.joint_pos[:, ids] - q_ref
    qd = robot.data.joint_vel[:, ids]
    root_lin_b = _root_lin_vel_b(robot)
    root_ang_b = _root_ang_vel_b(robot)
    g = projected_gravity(env, asset_name=asset_name, noise_std=0.0)
    torque = torch.zeros_like(q)
    for attr in ("applied_torque", "computed_torque", "joint_torque"):
        value = getattr(robot.data, attr, None)
        if isinstance(value, torch.Tensor):
            torque = value[:, ids]
            break
    return torch.cat((g, root_lin_b, root_ang_b, q, qd * 0.05, torque * float(torque_scale)), dim=-1)


def object_rel_full_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = _get_robot(env)
    root_pos, root_quat = _root_pos_quat(robot)
    obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = _object_pos_quat_vel_w(env)
    rel_pos_b = _quat_rotate_inverse(root_quat, obj_pos - root_pos)
    rel_lin_vel_b = _quat_rotate_inverse(root_quat, obj_lin_vel - _root_lin_vel_w(robot))
    rel_ang_vel_b = _quat_rotate_inverse(root_quat, obj_ang_vel - _root_ang_vel_w(robot))
    rel_quat = _quat_mul(_quat_conjugate(root_quat), obj_quat)
    return torch.cat((rel_pos_b, rel_quat, rel_lin_vel_b, rel_ang_vel_b), dim=-1)


def object_truth_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = _object_pos_quat_vel_w(env)
    visible_gt = state.tag_available.to(torch.float32).unsqueeze(-1)
    released = state.has_released.to(torch.float32).unsqueeze(-1)
    return torch.cat((obj_pos, obj_quat, obj_lin_vel, obj_ang_vel, visible_gt, released), dim=-1)


def root_state_privileged(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    robot = _get_robot(env, asset_name)
    root_pos, root_quat = _root_pos_quat(robot)
    return torch.cat((root_pos, root_quat, _root_lin_vel_w(robot), _root_ang_vel_w(robot)), dim=-1)


def hold_anchor_error(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, rel_lin_vel_b, obj_ang_vel_b = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    return torch.cat((err * float(scale), rel_lin_vel_b, obj_ang_vel_b), dim=-1)


def object_params(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    one_hot = mode_one_hot(env)
    release_step_s = state.release_step.to(torch.float32).unsqueeze(-1) * float(getattr(env, "step_dt", 0.02))
    return torch.cat(
        (
            state.object_size,
            state.object_mass,
            state.object_friction,
            state.object_restitution,
            state.obs_noise_scale,
            one_hot,
            state.release_velocity_w,
            release_step_s,
        ),
        dim=-1,
    )


def _sensor_force_tensor(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    n = _num_envs(env)
    device = _device(env)
    try:
        sensor = env.scene[sensor_name]
    except Exception:
        return torch.zeros(n, 3, device=device)
    data = getattr(sensor, "data", None)
    if data is None:
        return torch.zeros(n, 3, device=device)
    value = None
    for attr in ("net_forces_w", "force_matrix_w"):
        candidate = getattr(data, attr, None)
        if isinstance(candidate, torch.Tensor):
            value = candidate
            break
    hist = getattr(data, "net_forces_w_history", None)
    if isinstance(hist, torch.Tensor):
        value = hist[:, -1]
    if value is None:
        return torch.zeros(n, 3, device=device)
    value = value.to(device=device, dtype=torch.float32)
    # Sum over body/contact dimensions while preserving xyz.
    while value.dim() > 2:
        value = value.sum(dim=1)
    if value.shape[-1] != 3:
        value = value.reshape(n, -1, 3).sum(dim=1)
    return value


def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str], scale: float = 1.0) -> torch.Tensor:
    # Privileged critic-only term. It is intentionally not used by PolicyCfg.
    forces = [_sensor_force_tensor(env, name) for name in sensor_names]
    if not forces:
        return torch.zeros(_num_envs(env), 0, device=_device(env))
    out = torch.cat(forces, dim=-1) * float(scale)
    return torch.clamp(out, -10.0, 10.0)


# -----------------------------------------------------------------------------
# Helper functions used by rewards, terminations, and events.
# -----------------------------------------------------------------------------
def reaction_window(
    env: "ManagerBasedRLEnv",
    min_ttc: float = 0.08,
    max_ttc: float = 0.78,
    min_closing_speed: float = 0.24,
) -> torch.Tensor:
    """True only when the box is approaching and catch motion should begin soon."""
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    anchor = state.target_anchor_b
    distance_x = rel_pos_b[:, 0] - anchor[:, 0]
    lateral_err = torch.abs(rel_pos_b[:, 1] - anchor[:, 1])
    vertical_err = torch.abs(rel_pos_b[:, 2] - anchor[:, 2])
    closing_speed = -rel_vel_b[:, 0]
    ttc = distance_x / torch.clamp(closing_speed, min=1e-3)
    incoming = (distance_x > -0.08) & (closing_speed > min_closing_speed)
    roughly_catchable = (lateral_err < 0.55) & (vertical_err < 0.42)
    valid_type = (state.episode_type == EP_TOSS) | (state.episode_type == EP_DELAYED_TOSS)
    return incoming & roughly_catchable & valid_type & state.has_released & (ttc > min_ttc) & (ttc < max_ttc)


def near_catch_window(
    env: "ManagerBasedRLEnv",
    x_tol_front: float = 0.34,
    x_tol_back: float = 0.18,
    y_tol: float = 0.46,
    z_tol: float = 0.34,
) -> torch.Tensor:
    """Object is close enough that hug/hold rewards may be active.

    This is the anti-early-motion gate: v22 does not activate catch rewards merely
    because the object has been released.
    """
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    valid_type = (state.episode_type == EP_TOSS) | (state.episode_type == EP_DELAYED_TOSS)
    near = (err[:, 0] < float(x_tol_front)) & (err[:, 0] > -float(x_tol_back))
    near &= torch.abs(err[:, 1]) < float(y_tol)
    near &= torch.abs(err[:, 2]) < float(z_tol)
    return near & valid_type & state.has_released


def catchable_or_hold_phase(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return reaction_window(env) | near_catch_window(env) | hold_condition(env).squeeze(-1).to(torch.bool)


def post_catch_phase(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    in_pocket = (torch.abs(err[:, 0]) < 0.20) & (torch.abs(err[:, 1]) < 0.24) & (torch.abs(err[:, 2]) < 0.22)
    slowish = _safe_norm(rel_vel_b) < 0.65
    valid_type = (state.episode_type == EP_TOSS) | (state.episode_type == EP_DELAYED_TOSS)
    return state.has_released & valid_type & in_pocket & slowish


def hold_quality_terms(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, obj_ang_vel_b = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    qx = _exp_reward(err[:, 0], sigma=0.135)
    qy = _exp_reward(err[:, 1], sigma=0.165)
    qz = _exp_reward(err[:, 2], sigma=0.150)
    pos_quality = qx * qy * qz
    lin_quality = _exp_reward(_safe_norm(rel_vel_b), sigma=0.45)
    ang_quality = _exp_reward(_safe_norm(obj_ang_vel_b), sigma=1.45)
    z_ok = _exp_reward(rel_pos_b[:, 2] - state.target_anchor_b[:, 2], sigma=0.145)
    return pos_quality, lin_quality, ang_quality, z_ok


def chest_pocket_terms(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    x_quality = _exp_reward(err[:, 0], sigma=0.13)
    y_quality = _exp_reward(err[:, 1], sigma=0.16)
    z_quality = _exp_reward(err[:, 2], sigma=0.15)
    not_far_forward = torch.sigmoid((0.50 - rel_pos_b[:, 0]) * 12.0)
    return x_quality, y_quality, z_quality, not_far_forward


def hand_side_errors(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    left_b, right_b = _hand_positions_root_frame(env)
    half_y = 0.5 * state.object_size[:, 1]
    side_margin = 0.055
    x_back = -0.075
    z_bias = 0.000
    left_anchor = rel_pos_b + torch.stack(
        (torch.full_like(half_y, x_back), half_y + side_margin, torch.full_like(half_y, z_bias)), dim=-1
    )
    right_anchor = rel_pos_b + torch.stack(
        (torch.full_like(half_y, x_back), -(half_y + side_margin), torch.full_like(half_y, z_bias)), dim=-1
    )
    left_err = _safe_norm(left_b - left_anchor)
    right_err = _safe_norm(right_b - right_anchor)
    lateral_order = ((left_b[:, 1] - rel_pos_b[:, 1]) > 0.035) & ((right_b[:, 1] - rel_pos_b[:, 1]) < -0.035)
    left_depth = (left_b[:, 0] < rel_pos_b[:, 0] + 0.13) & (left_b[:, 0] > rel_pos_b[:, 0] - 0.34)
    right_depth = (right_b[:, 0] < rel_pos_b[:, 0] + 0.13) & (right_b[:, 0] > rel_pos_b[:, 0] - 0.34)
    depth_order = left_depth & right_depth
    return left_err, right_err, lateral_order.to(torch.float32), depth_order.to(torch.float32)


def hug_geometry_terms(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_err, right_err, lateral_order, depth_order = hand_side_errors(env)
    side_close = torch.exp(-(left_err * left_err + right_err * right_err) / (0.27 * 0.27))
    symmetry = torch.exp(-((left_err - right_err) ** 2) / (0.10 * 0.10))
    bracket = side_close * (0.35 + 0.65 * lateral_order) * (0.35 + 0.65 * depth_order)
    return bracket, symmetry, lateral_order, depth_order


def ready_pose_error(env: "ManagerBasedRLEnv", pose: dict[str, float] | None = None) -> torch.Tensor:
    if pose is None:
        pose = scene_objects_cfg.READY_POSE
    robot = _get_robot(env)
    ids = _controlled_joint_ids(env)
    q = robot.data.joint_pos[:, ids]
    q_ref = _controlled_pose_tensor(env, pose)
    return q - q_ref


__all__ = [
    "EP_TOSS",
    "EP_DELAYED_TOSS",
    "EP_NO_TOSS",
    "EP_PASS_BY",
    "EP_NEAR_MISS",
    "NUM_EPISODE_TYPES",
    "HOLD_ANCHOR_B",
    "get_task_state",
    "projected_gravity",
    "base_angular_velocity",
    "controlled_joint_pos_rel",
    "controlled_joint_velocities",
    "prev_actions",
    "object_rel_pos",
    "object_rel_lin_vel",
    "tag_visible",
    "mode_one_hot",
    "toss_state",
    "hold_state",
    "drop_state",
    "critic_robot_state",
    "object_rel_full_state",
    "object_truth_state",
    "root_state_privileged",
    "hold_anchor_error",
    "object_params",
    "contact_forces",
    "reaction_window",
    "near_catch_window",
    "catchable_or_hold_phase",
    "post_catch_phase",
    "hold_condition",
    "hold_quality_terms",
    "chest_pocket_terms",
    "hand_side_errors",
    "hug_geometry_terms",
    "ready_pose_error",
    "update_action_history",
    "_as_tensor",
    "_env_ids",
    "_episode_step",
    "_get_env_origins",
    "_get_object",
    "_get_robot",
    "_object_pos_quat_vel_w",
    "_object_rel_in_root_frame",
    "_quat_from_euler_xyz",
    "_quat_rotate",
    "_root_pos_quat",
    "_root_lin_vel_w",
    "_root_ang_vel_w",
    "_safe_norm",
]
