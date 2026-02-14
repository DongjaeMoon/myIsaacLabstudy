from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING
from .rewards import _get_stage
from .observations import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

def robot_fallen(env: "ManagerBasedRLEnv", min_root_z=0.55, min_upright=0.4) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    upright = (-g_b[:, 2])
    return (z < min_root_z) | (upright < min_upright)

def robot_fallen_degree(
    env: "ManagerBasedRLEnv",
    min_root_z: float = 0.55,
    max_tilt_deg: float = 66.4,   # <-- 여기 각도로 설정
) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w

    # world down vector in body frame
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)

    # upright = cos(tilt)
    upright = (-g_b[:, 2])

    # convert tilt threshold (deg) -> upright threshold (cos)
    upright_min = math.cos(math.radians(max_tilt_deg))

    return (z < min_root_z) | (upright < upright_min)


def object_dropped_curriculum(env: "ManagerBasedRLEnv", min_z=0.50, max_dist=3.0) -> torch.Tensor:
    # stage0에서는 drop을 의미있게 볼 필요 없어서 off
    if _get_stage(env) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    # --- [ADD] 던진 뒤에만 drop 판정 ---
    if hasattr(env, "_urop_toss_count"):
        active = (env._urop_toss_count > 0)
    else:
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    obj = env.scene["object"]
    robot = env.scene["robot"]

    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if torch.any(active):
        z = obj.data.root_pos_w[:, 2]
        dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
        drop = (z < min_z) | (dist > max_dist)
        out[active] = drop[active]
    return out

# terminations.py : 아래 함수 추가
def object_ground_contact_curriculum(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_object_ground",
    force_thr: float = 1.0,
) -> torch.Tensor:
    """Terminate immediately when the box touches the ground (contact sensor)."""
    # stage0에서는 object가 '비활성/대기'일 수 있으니 drop 체크를 끔(즉시 리셋 방지)
    if _get_stage(env) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    sensor = env.scene[sensor_name]
    f = sensor.data.net_forces_w  # shape can vary by IsaacLab version

    # robust reduction: take max norm over all extra dims except env dim
    mag = torch.linalg.norm(f, dim=-1)
    while mag.ndim > 1:
        mag = mag.max(dim=-1).values

    return mag > force_thr
