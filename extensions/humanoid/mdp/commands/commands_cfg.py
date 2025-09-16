# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .step_command import StepCommand
from .velocity_command import VelocityCommand

@configclass
class StepCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = StepCommand

    step_period: float = MISSING
    """Step period [env.step_dt]."""

    dstep_length: float | None = None
    """Desired step length [m]."""

    dstep_width: float | None = None
    """Desired step width [m]."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the step commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges | None = None
    """Distribution ranges for the velocity commands."""

@configclass
class VelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = VelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]

    reference: str = "body" # body or world
    """Reference frame in which the commands are expressed."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    lin_vel_threshold: float = 0.2
    """Threshold for linear velocity commands [m/s]."""

    ang_vel_threshold: float = 0.1
    """Threshold for angular velocity commands [rad/s]."""
