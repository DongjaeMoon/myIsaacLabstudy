# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass

@configclass
class UropG1LOCOv0PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "UROP_g1_loco_v0"
    
    # [핵심 수정 1] 주석 해제: State 정규화는 보행 학습의 필수 요소입니다.
    empirical_normalization = True 
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # 탐험을 조금 더 하도록 0.005 -> 0.01로 상향
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3, # [핵심 수정 2] 빠른 걸음걸이 학습을 위해 학습률 1.0e-3 적용
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0, 
    )
    clip_actions = 1.0