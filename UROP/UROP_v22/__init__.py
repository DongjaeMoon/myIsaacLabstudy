import gymnasium as gym

# Training environment: v22 randomized toss generalist.
gym.register(
    id="Isaac-Urop-v22",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v22_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV22PPORunnerCfg",
    },
)

# Play/evaluation environment.
gym.register(
    id="Isaac-Urop-v22-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v22_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV22PPORunnerCfg",
    },
)


# Demo environment: single clean front-toss for video recording.
gym.register(
    id="Isaac-Urop-v22-Demo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v22_EnvCfg_Demo",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV22PPORunnerCfg",
    },
)
