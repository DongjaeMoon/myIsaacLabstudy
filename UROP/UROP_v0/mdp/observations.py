# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def robot_proprio(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N, D) proprio: joint pos/vel + base lin/ang vel + base orientation(quat)."""
    robot = env.scene["robot"]
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel
    base_lin = robot.data.root_lin_vel_w
    base_ang = robot.data.root_ang_vel_w
    base_quat = robot.data.root_quat_w
    return torch.cat([jp, jv, base_lin, base_ang, base_quat], dim=-1)

def object_rel_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N, 6) object relative pos/vel wrt robot root."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rel_pos = obj.data.root_pos_w - robot.data.root_pos_w
    rel_vel = obj.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return torch.cat([rel_pos, rel_vel], dim=-1)


def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str]) -> torch.Tensor:
    outs = []
    for name in sensor_names:
        sensor = env.scene[name]
        f = sensor.data.net_forces_w  # (N, 1, 3)일 가능성 높음 (링크 1개라서)
        outs.append(f.reshape(f.shape[0], -1))
    return torch.cat(outs, dim=-1)
