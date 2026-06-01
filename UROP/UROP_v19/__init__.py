import gymnasium as gym

# Training environment: randomized handover/carry-prep MDP.
gym.register(
    id="Isaac-Urop-v19",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v19_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV19PPORunnerCfg",
    },
)

# Mixed GUI/evaluation environment. Keeps the random handover/non-handover mixture.
gym.register(
    id="Isaac-Urop-v19-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v19_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV19PPORunnerCfg",
    },
)

# Debug GUI environment. Prints sampled task/mass/visibility statistics.
gym.register(
    id="Isaac-Urop-v19-Debug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v19_EnvCfg_Debug",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV19PPORunnerCfg",
    },
)
