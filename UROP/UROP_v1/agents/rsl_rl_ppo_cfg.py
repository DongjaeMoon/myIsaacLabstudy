# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class SteelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 50
    experiment_name = "UROP_v1"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        # num_mini_batches=8,
        learning_rate=5.0e-4, #5.0e-4->1.0e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0, #1.0->0.5
    )
     # (4) action clip을 여기서 강제로 걸어주기 (train_rsl_rl.py가 agent_cfg.clip_actions를 씀)
    clip_actions = 1.0  # [-1,1]로 자르기(보통 float max abs로 씀)
