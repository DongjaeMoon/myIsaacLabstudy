import gymnasium as gym

# Training environment: v27 low-arc, low-gain, stable-stance catching task.
gym.register(
    id="Isaac-Urop-v27",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v27_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV27PPORunnerCfg",
    },
)

# Easier stochastic evaluation.
gym.register(
    id="Isaac-Urop-v27-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v27_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV27PPORunnerCfg",
    },
)

# Clean single-env video/demo task.
gym.register(
    id="Isaac-Urop-v27-Demo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v27_EnvCfg_Demo",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV27PPORunnerCfg",
    },
)
