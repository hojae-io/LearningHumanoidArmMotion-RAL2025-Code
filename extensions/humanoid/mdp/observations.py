# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
Root state.
"""

def base_lin_vel_world(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w

def base_heading(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root heading in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(-1)


"""
Joint state.
"""


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]

def centroidal_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal momentum of the asset."""
    return env.CM

def centroidal_momentum_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal momentum of the asset in the base frame."""
    return env.CM_bf

def centroidal_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des

def centroidal_momentum_des_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset in the base frame."""
    return env.CM_des_bf

def centroidal_linear_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal linear momentum of the asset."""
    return env.CM[:, :3]

def centroidal_linear_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des[:, :3]

def centroidal_angular_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset."""
    return env.CM[:, 3:]

def centroidal_angular_momentum_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset in the base frame."""
    return env.CM_bf[:, 3:]

def centroidal_angular_momentum_mixed(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset in the mixed frame. \
       x, y axis in the base frame, z axis in the world frame."""
    return torch.hstack([env.CM_bf[:, 3:5], env.CM[:, 5:6]])

def centroidal_angular_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des[:, 3:]

def centroidal_angular_momentum_des_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset in the base frame."""
    return env.CM_des_bf[:, 3:]

def centroidal_angular_momentum_des_mixed(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal angular momentum of the asset in the mixed frame. \
       x, y axis in the base frame, z axis in the world frame."""
    return torch.hstack([env.CM_des_bf[:, 3:5], env.CM_des[:, 5:6]])

def centroidal_yaw_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal momentum of the asset."""
    return env.CM[:, 5:6]

def centroidal_yaw_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des[:, 5:6]

"""
Robot state.
"""

def foot_states_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The right foot states relative to the base frame."""
    
    return env.foot_states_right

def foot_states_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The left foot states relative to the base frame."""
    
    return env.foot_states_left

"""
Commands.
"""

def phase_sin(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sin of the phase variable."""
    return env.phase_sin

def phase_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cos of the phase variable."""
    return env.phase_cos

def joint_effort_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The target joint efforts."""
    return env.joint_effort_target.float()

def step_commands_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The right foot step commands relative to the base frame."""
    return env.step_commands_right

def step_commands_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The left foot step commands relative to the base frame."""
    return env.step_commands_left
