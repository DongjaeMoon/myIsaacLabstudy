import gymnasium as gym

# Carry-only training environment.
# Reset starts from a bank of successful catch states.
gym.register(
    id="Isaac-Urop-carry-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_carry_v2_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropCarryV2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Urop-carry-v2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_carry_v2_EnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropCarryV2PPORunnerCfg",
    },
)
