import gymnasium as gym

# Training environment: v20 randomized toss generalist.
gym.register(
    id="Isaac-Urop-v20",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v20_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV20PPORunnerCfg",
    },
)

# Play/evaluation environment.
gym.register(
    id="Isaac-Urop-v20-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v20_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV20PPORunnerCfg",
    },
)
