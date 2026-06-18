import gymnasium as gym

# Training environment: v23 low-arc, stable-stance catching task.
gym.register(
    id="Isaac-Urop-v23",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v23_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV23PPORunnerCfg",
    },
)

# Easier stochastic evaluation.
gym.register(
    id="Isaac-Urop-v23-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v23_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV23PPORunnerCfg",
    },
)

# Clean single-env video/demo task.
gym.register(
    id="Isaac-Urop-v23-Demo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v23_EnvCfg_Demo",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV23PPORunnerCfg",
    },
)
