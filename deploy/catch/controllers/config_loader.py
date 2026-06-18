# [/home/<user>/.../myIsaacLabstudy/deploy/catch/controllers/config_loader.py]
"""Config helpers for Isaac-Sim deployment controllers.

This file intentionally keeps the old public functions used by v8/v12 code, but
makes local filesystem paths robust.  Isaac Sim extensions often run with a CWD
that is not ``myIsaacLabstudy/`` and ``omni.client.read_file`` may behave
differently for plain local paths, so we first try normal Python IO and then fall
back to omni.client.
"""
from __future__ import annotations

import fnmatch
import io
import os
import sys
from typing import List

import numpy as np
import yaml

try:
    import omni.client  # type: ignore
except Exception:  # Allows syntax checks outside Isaac Sim.
    omni = None  # type: ignore


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    """YAML loader that ignores Isaac Lab python-object/function tags."""


def _tuple_constructor(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


def _ignore_unknown(loader: yaml.Loader, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return None


SafeLoaderIgnoreUnknown.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_constructor)
SafeLoaderIgnoreUnknown.add_constructor(None, _ignore_unknown)


def _read_bytes(path: str) -> bytes:
    """Read a local or omni-client path as bytes."""
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.exists(expanded):
        with open(expanded, "rb") as f:
            return f.read()

    if omni is not None:
        result = omni.client.read_file(expanded)  # type: ignore[attr-defined]
        # Isaac Sim versions differ: some return (Result, version, content),
        # others return a tuple-like object.  The content is the last item.
        content = result[-1]
        return memoryview(content).tobytes()

    raise FileNotFoundError(f"Could not read config path: {path}")


def parse_env_config(env_config_path: str = "env.yaml") -> dict:
    raw = _read_bytes(env_config_path)
    data = yaml.load(io.BytesIO(raw), Loader=SafeLoaderIgnoreUnknown)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid env.yaml content at {env_config_path!r}")
    return data


def _pattern_matches(joint_name: str, pattern: str) -> bool:
    # Isaac Lab configs often use exact names, regex-like ".*", or glob-like "*".
    glob = str(pattern).replace(".*", "*")
    if fnmatch.fnmatch(joint_name, glob):
        return True
    # Old deploy code appended "*".  Keep that behavior for patterns that are
    # prefixes in some YAML exports.
    return fnmatch.fnmatch(joint_name, glob + "*")


def _expand_pattern_dict(value, joint_names_expr) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            if v is not None:
                out[str(k)] = float(v)
        return out
    if isinstance(value, (float, int)):
        for pattern in joint_names_expr:
            out[str(pattern)] = float(value)
    return out


def get_robot_joint_properties(data: dict, joint_names: List[str]):
    robot_cfg = data.get("scene", {}).get("robot", {})
    actuator_data = robot_cfg.get("actuators", {}) or {}

    stiffness_dict: dict[str, float] = {}
    damping_dict: dict[str, float] = {}
    effort_dict: dict[str, float] = {}
    velocity_dict: dict[str, float] = {}
    armature_dict: dict[str, float] = {}

    for actuator in actuator_data.values():
        joint_names_expr = actuator.get("joint_names_expr", []) or []
        stiffness_dict.update(_expand_pattern_dict(actuator.get("stiffness"), joint_names_expr))
        damping_dict.update(_expand_pattern_dict(actuator.get("damping"), joint_names_expr))
        armature_dict.update(_expand_pattern_dict(actuator.get("armature"), joint_names_expr))

        effort_limit = actuator.get("effort_limit_sim", actuator.get("effort_limit"))
        velocity_limit = actuator.get("velocity_limit_sim", actuator.get("velocity_limit"))
        if isinstance(effort_limit, dict):
            effort_dict.update(_expand_pattern_dict(effort_limit, []))
        elif isinstance(effort_limit, (float, int)):
            for pattern in joint_names_expr:
                effort_dict[str(pattern)] = float(effort_limit)
        if isinstance(velocity_limit, dict):
            velocity_dict.update(_expand_pattern_dict(velocity_limit, []))
        elif isinstance(velocity_limit, (float, int)):
            for pattern in joint_names_expr:
                velocity_dict[str(pattern)] = float(velocity_limit)

    init_state = robot_cfg.get("init_state", {}) or {}
    default_pos_dict: dict[str, float] = {}
    default_vel_dict: dict[str, float] = {}

    init_joint_pos = init_state.get("joint_pos", {})
    if isinstance(init_joint_pos, dict):
        default_pos_dict.update({str(k): float(v) for k, v in init_joint_pos.items()})
    elif isinstance(init_joint_pos, (float, int)):
        default_pos_dict["*"] = float(init_joint_pos)

    init_joint_vel = init_state.get("joint_vel", {})
    if isinstance(init_joint_vel, dict):
        default_vel_dict.update({str(k): float(v) for k, v in init_joint_vel.items()})
    elif isinstance(init_joint_vel, (float, int)):
        default_vel_dict["*"] = float(init_joint_vel)

    def find_val(joint: str, val_dict: dict[str, float], default_val: float) -> float:
        # Prefer exact key over pattern match.
        if joint in val_dict:
            return float(val_dict[joint])
        for pattern, pattern_val in val_dict.items():
            if _pattern_matches(joint, pattern):
                return float(pattern_val)
        return float(default_val)

    stiffness_inorder = []
    damping_inorder = []
    effort_limits_inorder = []
    velocity_limits_inorder = []
    armature_inorder = []
    default_pos_inorder = []
    default_vel_inorder = []

    for joint in joint_names:
        stiffness_inorder.append(find_val(joint, stiffness_dict, 0.0))
        damping_inorder.append(find_val(joint, damping_dict, 0.0))
        effort_limits_inorder.append(find_val(joint, effort_dict, float(sys.maxsize)))
        velocity_limits_inorder.append(find_val(joint, velocity_dict, float(sys.maxsize)))
        armature_inorder.append(find_val(joint, armature_dict, 0.0))
        default_pos_inorder.append(find_val(joint, default_pos_dict, 0.0))
        default_vel_inorder.append(find_val(joint, default_vel_dict, 0.0))

    return (
        np.array(effort_limits_inorder, dtype=np.float32),
        np.array(velocity_limits_inorder, dtype=np.float32),
        np.array(stiffness_inorder, dtype=np.float32),
        np.array(damping_inorder, dtype=np.float32),
        np.array(armature_inorder, dtype=np.float32),
        np.array(default_pos_inorder, dtype=np.float32),
        np.array(default_vel_inorder, dtype=np.float32),
    )


def get_articulation_props(data: dict) -> dict:
    return data.get("scene", {}).get("robot", {}).get("spawn", {}).get("articulation_props") or {}


def get_physics_properties(data: dict):
    decimation = int(data.get("decimation", 1))
    sim_cfg = data.get("sim", {}) or {}
    dt = float(sim_cfg.get("dt", 1.0 / 60.0))
    render_interval = int(sim_cfg.get("render_interval", decimation))
    return decimation, dt, render_interval
