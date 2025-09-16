# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import pendulum_runner_cfg, pendulum_task_cfg

from .pendulum_env import PendulumEnv

##
# Register Gym environments.
##

gym.register(
    id="pendulum",
    entry_point="isaaclab.envs:PendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pendulum_task_cfg.PendulumEnvCfg,
        "rsl_rl_cfg_entry_point": pendulum_runner_cfg.PendulumPPORunnerCfg,
    },
)

gym.register(
    id="pendulum_play",
    entry_point="isaaclab.envs:PendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pendulum_task_cfg.PendulumEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pendulum_runner_cfg.PendulumPPORunnerCfg,
    },
)
