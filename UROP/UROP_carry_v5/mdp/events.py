# [/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v5/mdp/events.py]

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from .observations import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = (0.32, 0.24, 0.24)

# upper-body grasp preservation
ARM_REF_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

WRIST_REF_JOINT_NAMES = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

UPPER_REF_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# lower-body stance preservation
LEG_REF_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def _get_joint_name_indices(env: "ManagerBasedRLEnv", joint_names: list[str], cache_attr: str) -> torch.Tensor:
    if hasattr(env, cache_attr):
        return getattr(env, cache_attr)

    robot = env.scene["robot"]
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}

    missing = [n for n in joint_names if n not in name_to_idx]
    if len(missing) > 0:
        raise RuntimeError(f"Missing joints for carry reference buffer: {missing}")

    idx = torch.tensor([name_to_idx[n] for n in joint_names], device=env.device, dtype=torch.long)
    setattr(env, cache_attr, idx)
    return idx


def _ensure_carry_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device

    if not hasattr(env, "_carry_bank_loaded"):
        env._carry_bank_loaded = False

    if not hasattr(env, "_carry_bank"):
        env._carry_bank = None

    if not hasattr(env, "_carry_bank_size"):
        env._carry_bank_size = 0

    if not hasattr(env, "_carry_reset_grace_steps"):
        env._carry_reset_grace_steps = torch.zeros(n, dtype=torch.int32, device=d)

    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), 3.0, device=d)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.7, device=d)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.02, device=d)

    if not hasattr(env, "_carry_target_obj_rel"):
        env._carry_target_obj_rel = torch.tensor([0.42, 0.0, 0.22], device=d).unsqueeze(0).repeat(n, 1)

    if not hasattr(env, "_carry_ref_arm_joint_pos"):
        env._carry_ref_arm_joint_pos = torch.zeros((n, len(ARM_REF_JOINT_NAMES)), device=d)

    if not hasattr(env, "_carry_ref_wrist_joint_pos"):
        env._carry_ref_wrist_joint_pos = torch.zeros((n, len(WRIST_REF_JOINT_NAMES)), device=d)

    if not hasattr(env, "_carry_ref_upper_joint_pos"):
        env._carry_ref_upper_joint_pos = torch.zeros((n, len(UPPER_REF_JOINT_NAMES)), device=d)

    if not hasattr(env, "_carry_ref_leg_joint_pos"):
        env._carry_ref_leg_joint_pos = torch.zeros((n, len(LEG_REF_JOINT_NAMES)), device=d)

    if not hasattr(env, "_carry_ref_root_height"):
        env._carry_ref_root_height = torch.full((n,), 0.79, device=d)


def _load_carry_bank(env: "ManagerBasedRLEnv", bank_path: str) -> None:
    _ensure_carry_buffers(env)

    if env._carry_bank_loaded:
        return

    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Carry bank file not found: {bank_path}")

    bank = torch.load(bank_path, map_location="cpu")

    required = ["root_pose", "root_vel", "joint_pos", "joint_vel", "object_pose", "object_vel"]
    for k in required:
        if k not in bank:
            raise KeyError(f"Missing key '{k}' in carry bank: {bank_path}")

    n = int(bank["root_pose"].shape[0])
    if n <= 0:
        raise RuntimeError(f"Carry bank is empty: {bank_path}")

    env._carry_bank = bank
    env._carry_bank_size = n
    env._carry_bank_loaded = True


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
    mass_range=(2.0, 4.5),
    friction_range=(0.55, 0.95),
    restitution_range=(0.00, 0.06),
    apply_physx: bool = True,
) -> None:
    _ensure_carry_buffers(env)

    d = env.device
    n = int(env_ids.shape[0])
    if n == 0:
        return

    mass = torch.empty((n, 1), device=d).uniform_(*mass_range)
    friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)

    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)

    if apply_physx:
        _apply_physx_mass_material_best_effort(env, env_ids, mass, friction, restitution)


def reset_from_catch_success_bank(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    bank_path: str,
    pos_noise_xy: float = 0.002,
    yaw_noise_rad: float = 0.0,   # intentionally unused for now
    vel_noise_scale: float = 0.0,
    grace_steps: int = 10,
    randomize_object: bool = False,
) -> None:
    """Reset carry env from catch-success bank.

    carry_v5 assumption:
      catch already solved grasp + stabilize.
      carry only has to preserve that posture and recover locomotion.
    """
    _ensure_carry_buffers(env)
    _load_carry_bank(env, bank_path)

    if int(env_ids.numel()) == 0:
        return

    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    n = int(env_ids.shape[0])

    bank_ids = torch.randint(0, env._carry_bank_size, (n,), device=d)
    bank_ids_cpu = bank_ids.cpu()
    bank = env._carry_bank

    root_pose_local = bank["root_pose"][bank_ids_cpu].to(d).clone()
    root_vel = bank["root_vel"][bank_ids_cpu].to(d).clone()
    joint_pos = bank["joint_pos"][bank_ids_cpu].to(d).clone()
    joint_vel = bank["joint_vel"][bank_ids_cpu].to(d).clone()
    object_pose_local = bank["object_pose"][bank_ids_cpu].to(d).clone()
    object_vel = bank["object_vel"][bank_ids_cpu].to(d).clone()

    arm_ref_idx = _get_joint_name_indices(env, ARM_REF_JOINT_NAMES, "_carry_ref_arm_joint_indices")
    wrist_ref_idx = _get_joint_name_indices(env, WRIST_REF_JOINT_NAMES, "_carry_ref_wrist_joint_indices")
    upper_ref_idx = _get_joint_name_indices(env, UPPER_REF_JOINT_NAMES, "_carry_ref_upper_joint_indices")
    leg_ref_idx = _get_joint_name_indices(env, LEG_REF_JOINT_NAMES, "_carry_ref_leg_joint_indices")

    env_origins = env.scene.env_origins[env_ids]

    root_pose_world = root_pose_local.clone()
    root_pose_world[:, 0:3] += env_origins

    object_pose_world = object_pose_local.clone()
    object_pose_world[:, 0:3] += env_origins

    if pos_noise_xy > 0.0:
        root_pose_world[:, 0] += torch.empty(n, device=d).uniform_(-pos_noise_xy, pos_noise_xy)
        root_pose_world[:, 1] += torch.empty(n, device=d).uniform_(-pos_noise_xy, pos_noise_xy)
        object_pose_world[:, 0] += torch.empty(n, device=d).uniform_(-pos_noise_xy, pos_noise_xy)
        object_pose_world[:, 1] += torch.empty(n, device=d).uniform_(-pos_noise_xy, pos_noise_xy)

    if vel_noise_scale > 0.0:
        root_vel += vel_noise_scale * torch.randn_like(root_vel)
        object_vel += vel_noise_scale * torch.randn_like(object_vel)
        joint_vel += vel_noise_scale * torch.randn_like(joint_vel)

    robot.write_root_pose_to_sim(root_pose_world, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    obj.write_root_pose_to_sim(object_pose_world, env_ids=env_ids)
    obj.write_root_velocity_to_sim(object_vel, env_ids=env_ids)

    if randomize_object:
        randomize_carry_object(env, env_ids, apply_physx=True)

    # reference poses from THIS sampled catch-stabilized state
    env._carry_ref_arm_joint_pos[env_ids] = joint_pos[:, arm_ref_idx]
    env._carry_ref_wrist_joint_pos[env_ids] = joint_pos[:, wrist_ref_idx]
    env._carry_ref_upper_joint_pos[env_ids] = joint_pos[:, upper_ref_idx]
    env._carry_ref_leg_joint_pos[env_ids] = joint_pos[:, leg_ref_idx]
    env._carry_ref_root_height[env_ids] = root_pose_world[:, 2]

    env._carry_reset_grace_steps[env_ids] = int(grace_steps)

    rel_now_body = quat_rotate_inverse(
        root_pose_world[:, 3:7],
        object_pose_world[:, 0:3] - root_pose_world[:, 0:3],
    )
    env._carry_target_obj_rel[env_ids] = rel_now_body


def hold_upper_body_reference(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Keep waist+arms+wrists at the catch-stabilized reference pose every step."""
    _ensure_carry_buffers(env)

    if int(env_ids.numel()) == 0:
        return

    robot = env.scene["robot"]
    upper_idx = _get_joint_name_indices(env, UPPER_REF_JOINT_NAMES, "_carry_ref_upper_joint_indices")
    q_ref = env._carry_ref_upper_joint_pos[env_ids]

    robot.set_joint_position_target(q_ref, joint_ids=upper_idx, env_ids=env_ids)


def decay_reset_grace(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    _ensure_carry_buffers(env)
    if int(env_ids.numel()) == 0:
        return
    env._carry_reset_grace_steps[env_ids] = torch.clamp(env._carry_reset_grace_steps[env_ids] - 1, min=0)