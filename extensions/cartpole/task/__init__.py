# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import cartpole_runner_cfg, cartpole_task_cfg

from .cartpole_env import CartpoleEnv

##
# Register Gym environments.
##

gym.register(
    id="cartpole",
    entry_point="isaaclab.envs:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cartpole_task_cfg.CartpoleEnvCfg,
        "rsl_rl_cfg_entry_point": cartpole_runner_cfg.CartpolePPORunnerCfg,
    },
)

gym.register(
    id="cartpole_play",
    entry_point="isaaclab.envs:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cartpole_task_cfg.CartpoleEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": cartpole_runner_cfg.CartpolePPORunnerCfg,
    },
)
