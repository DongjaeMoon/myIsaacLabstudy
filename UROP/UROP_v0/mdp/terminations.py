# UROP/UROP_v0/mdp/terminations.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_fallen(env: "ManagerBasedRLEnv", min_root_z: float = 0.55) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, 2] < min_root_z


def object_dropped(env: "ManagerBasedRLEnv", min_z: float = 0.2) -> torch.Tensor:
    obj = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < min_z
