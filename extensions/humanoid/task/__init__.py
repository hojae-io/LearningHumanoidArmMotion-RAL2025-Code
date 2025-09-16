# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import humanoid_vanilla_runner_cfg, humanoid_vanilla_task_cfg
from . import humanoid_full_vanilla_runner_cfg, humanoid_full_vanilla_task_cfg
from . import humanoid_full_modular_runner_cfg, humanoid_full_modular_task_cfg

from .humanoid_vanilla import HumanoidVanillaEnv
from .humanoid_full_vanilla import HumanoidFullVanillaEnv
from .humanoid_full_modular import HumanoidFullModularEnv

##
# Register Gym environments.
##

gym.register(
    id="humanoid_vanilla",
    entry_point=humanoid_vanilla.HumanoidVanillaEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanoid_vanilla_task_cfg.HumanoidVanillaEnvCfg,
        "rsl_rl_cfg_entry_point": humanoid_vanilla_runner_cfg.HumanoidVanillaPPORunnerCfg,
    },
)

gym.register(
    id="humanoid_vanilla_play",
    entry_point=humanoid_vanilla.HumanoidVanillaEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanoid_vanilla_task_cfg.HumanoidVanillaEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": humanoid_vanilla_runner_cfg.HumanoidVanillaPPORunnerCfg,
    },
)

gym.register(
    id="humanoid_full_vanilla",
    entry_point=humanoid_full_vanilla.HumanoidFullVanillaEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanoid_full_vanilla_task_cfg.HumanoidFullVanillaEnvCfg,
        "rsl_rl_cfg_entry_point": humanoid_full_vanilla_runner_cfg.HumanoidFullVanillaPPORunnerCfg,
    },
)

gym.register(
    id="humanoid_full_vanilla_play",
    entry_point=humanoid_full_vanilla.HumanoidFullVanillaEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanoid_full_vanilla_task_cfg.HumanoidFullVanillaEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": humanoid_full_vanilla_runner_cfg.HumanoidFullVanillaPPORunnerCfg,
    },
)

gym.register(
    id="humanoid_full_modular",
    entry_point=humanoid_full_modular.HumanoidFullModularEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanoid_full_modular_task_cfg.HumanoidFullModularEnvCfg,
        "rsl_rl_cfg_entry_point": humanoid_full_modular_runner_cfg.HumanoidFullModularPPORunnerCfg,
    },
)

gym.register(
    id="humanoid_full_modular_play",
    entry_point=humanoid_full_modular.HumanoidFullModularEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanoid_full_modular_task_cfg.HumanoidFullModularEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": humanoid_full_modular_runner_cfg.HumanoidFullModularPPORunnerCfg,
    },
)
