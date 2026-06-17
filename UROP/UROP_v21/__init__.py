import gymnasium as gym

# Training environment: v21 randomized toss generalist.
gym.register(
    id="Isaac-Urop-v21",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v21_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV21PPORunnerCfg",
    },
)

# Play/evaluation environment.
gym.register(
    id="Isaac-Urop-v21-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v21_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV21PPORunnerCfg",
    },
)
