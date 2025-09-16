"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_velocity_command_range(env: ManagerBasedRLEnv, term_name: str, num_steps: int):
    """ Curriculums that modifies the velocity command range a given number of steps.

    Args:
        env: The learning environment.
        term_name: The name of the command term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.command_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.ranges.lin_vel_x = (0.1, 0.2)
        term_cfg.ranges.lin_vel_y = (0.1, 0.2)
        term_cfg.ranges.ang_vel_z = (0.1, 0.2)
        env.command_manager.set_term_cfg(term_name, term_cfg)