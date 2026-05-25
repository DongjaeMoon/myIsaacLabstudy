import gymnasium as gym

# [v16] 학습용 환경 (커리큘럼 자동 진행)
gym.register(
    id="Isaac-Urop-v16",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v16_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV16PPORunnerCfg",
    },
)

# [v16] 평가(Play)용 환경
gym.register(
    id="Isaac-Urop-v16-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v16_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV16PPORunnerCfg",
    },
)
