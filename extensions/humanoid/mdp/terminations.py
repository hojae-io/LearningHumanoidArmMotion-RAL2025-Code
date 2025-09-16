# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


"""
Root terminations.
"""

def base_termination(
    env: ManagerBasedRLEnv, max_lin_vel: float = None, max_ang_vel: float = None, max_tilting: float = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")                
) -> torch.Tensor:
    """Terminate when the asset's linear velocity exceeds the maximum linear velocity."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if max_lin_vel is not None:
        terminated |= torch.any(torch.norm(asset.data.root_lin_vel_b, dim=-1, keepdim=True) > max_lin_vel, dim=1)
    if max_ang_vel is not None:
        terminated |= torch.any(torch.norm(asset.data.root_ang_vel_b, dim=-1, keepdim=True) > max_ang_vel, dim=1)
    if max_tilting is not None:
        terminated |= torch.any(torch.abs(asset.data.projected_gravity_b[:, 0:1]) > max_tilting, dim=1)
        terminated |= torch.any(torch.abs(asset.data.projected_gravity_b[:, 1:2]) > max_tilting, dim=1)
    return terminated


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # check if any contact force exceeds the threshold
    term_contact = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=-1)
    terminated = torch.any((term_contact > threshold), dim=1)
    return terminated