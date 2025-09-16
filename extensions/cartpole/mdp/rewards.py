# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REWARD_TRACKING_SIGMA = 0.25

def joint_torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.applied_torque), dim=1)

def cart_vel(env: ManagerBasedRLEnv):
    asset: Articulation = env.scene['robot']
    return -torch.square(asset.data.joint_vel[:, 0]) * env.pendulum_is_upright

def pendulum_vel(env: ManagerBasedRLEnv):
    asset: Articulation = env.scene['robot']
    return -torch.square(asset.data.joint_vel[:, 1]) * env.pendulum_is_upright

def upright_pendulum(env: ManagerBasedRLEnv):
    asset: Articulation = env.scene['robot']
    return torch.exp(-torch.square(asset.data.joint_pos[:, 1]) / REWARD_TRACKING_SIGMA)

