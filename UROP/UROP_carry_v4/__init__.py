import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Urop-carry-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "UROP_carry_v4.env_cfg:dj_urop_carry_v4_EnvCfg",
        "rsl_rl_cfg_entry_point": "UROP_carry_v4.agents.rsl_rl_ppo_cfg:UropCarryv4PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Urop-carry-v4-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "UROP_carry_v4.env_cfg:dj_urop_carry_v4_EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "UROP_carry_v4.agents.rsl_rl_ppo_cfg:UropCarryv4PPORunnerCfg",
    },
)
