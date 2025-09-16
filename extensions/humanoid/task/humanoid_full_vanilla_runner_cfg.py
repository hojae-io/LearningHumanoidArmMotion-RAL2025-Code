# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HumanoidFullVanillaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42 # -1
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 200
    experiment_name = "Humanoid_Full_Vanilla"
    logger = "wandb"
    enable_logging = True
    wandb_project = "Humanoid_Full_Vanilla"
    store_code_state = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        normalize_obs=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12143, # 0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.00041, # 1.e-5,
        schedule="adaptive",
        gamma=0.9751, # 0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.,
    )
