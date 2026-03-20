import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Urop-carry-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "UROP_carry_v5.env_cfg:dj_urop_carry_v5_EnvCfg",
        "rsl_rl_cfg_entry_point": "UROP_carry_v5.agents.rsl_rl_ppo_cfg:UropCarryv5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Urop-carry-v5-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "UROP_carry_v5.env_cfg:dj_urop_carry_v5_EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "UROP_carry_v5.agents.rsl_rl_ppo_cfg:UropCarryv5PPORunnerCfg",
    },
)
