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
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# * --- Regularization Rewards --- * #

def action_smoothness1(env: ManagerBasedRLEnv, action_name: str):
    action = env.action_manager.action[action_name]
    prev_action = env.action_manager.prev_action[action_name]
    dt2 = (env.step_dt*env.cfg.decimation)**2
    error = torch.square(action - prev_action)/dt2
    return -torch.sum(error, dim=1)

def action_smoothness2(env: ManagerBasedRLEnv, action_name: str):
    action = env.action_manager.action[action_name]
    prev_action = env.action_manager.prev_action[action_name]
    prev_prev_action = env.action_manager.prev_prev_action[action_name]
    dt2 = (env.step_dt*env.cfg.decimation)**2
    error = torch.square(action - 2*prev_action + prev_prev_action)/dt2
    return -torch.sum(error, dim=1)

def joint_position_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return -torch.sum(torch.square(error), dim=1) * _negsqrd_exp(torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1), scale=1.)

def joint_torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.applied_torque[:,asset_cfg.joint_ids]), dim=1)

def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.joint_vel[:,asset_cfg.joint_ids]), dim=1)

def base_lin_vel_z_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    return -torch.square(asset.data.root_lin_vel_b[:, 2])

def base_ang_vel_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def joint_pos_limits_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize joint positions too close to the limit
    out_of_limits = -(asset.data.joint_pos - asset.data.soft_joint_pos_limits[:,:,0]).clip(max=0.) # lower limit
    out_of_limits += (asset.data.joint_pos - asset.data.soft_joint_pos_limits[:,:,1]).clip(min=0.)
    return -torch.sum(out_of_limits[:,asset_cfg.joint_ids], dim=1)

def joint_torque_limits_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize joint torques too close to the limit
    out_of_limits = (torch.abs(asset.data.applied_torque) - asset.data.soft_joint_torque_limits).clip(min=0.)
    return -torch.sum(out_of_limits[:,asset_cfg.joint_ids], dim=1)

def joint_regularization_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Reward joint poses and symmetry
    error = 0.

    # Yaw joints regularization around 0
    right_hip_yaw_idx = asset.data.joint_names.index('a01_right_hip_yaw')
    left_hip_yaw_idx = asset.data.joint_names.index('a06_left_hip_yaw')
    error += _negsqrd_exp(asset.data.joint_pos[:, right_hip_yaw_idx] - asset.data.default_joint_pos[:, right_hip_yaw_idx])
    error += _negsqrd_exp(asset.data.joint_pos[:, left_hip_yaw_idx] - asset.data.default_joint_pos[:, left_hip_yaw_idx])
        
    # Ab/ad joint symmetry
    # error += self._negsqrd_exp(
    #     (self.dof_pos[:, 1] - self.dof_pos[:, 6])
    #     / self.scales['dof_pos'])
    right_hip_abad_idx = asset.data.joint_names.index('a02_right_hip_abad')
    left_hip_abad_idx = asset.data.joint_names.index('a07_left_hip_abad')
    error += _negsqrd_exp(asset.data.joint_pos[:, right_hip_abad_idx] - asset.data.default_joint_pos[:, right_hip_abad_idx])
    error += _negsqrd_exp(asset.data.joint_pos[:, left_hip_abad_idx] - asset.data.default_joint_pos[:, left_hip_abad_idx])

    # Pitch joint symmetry
    # error += self._negsqrd_exp(
    #     (self.dof_pos[:, 2] + self.dof_pos[:, 7])
    #     / self.scales['dof_pos'])
    # error += self._negsqrd_exp((self.dof_pos[:, 3] - self.dof_pos[:, 8]) / self.scales['dof_pos']) # knee

    return error/4

def arm_joint_regularization_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Reward joint poses and symmetry
    error = 0.

    # # Shoulder pitch symmetry
    right_shoulder_pitch_idx = asset.data.joint_names.index('a11_right_shoulder_pitch')
    left_shoulder_pitch_idx = asset.data.joint_names.index('a15_left_shoulder_pitch')
    error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_pitch_idx])
    error += _negsqrd_exp(asset.data.joint_pos[:, left_shoulder_pitch_idx])
    
    # # Shoulder ab/ad regularization around 0
    right_shoulder_abad_idx = asset.data.joint_names.index('a12_right_shoulder_abad')
    left_shoulder_abad_idx = asset.data.joint_names.index('a16_left_shoulder_abad')
    error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_abad_idx])
    error += _negsqrd_exp(asset.data.joint_pos[:, left_shoulder_abad_idx])

    # # Shoulder yaw regularization around 0
    right_shoulder_yaw_idx = asset.data.joint_names.index('a13_right_shoulder_yaw')
    left_shoulder_yaw_idx = asset.data.joint_names.index('a17_left_shoulder_yaw')
    error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_yaw_idx])
    error += _negsqrd_exp(asset.data.joint_pos[:, left_shoulder_yaw_idx])

    # # Elbow regularization around 0
    right_elbow_idx = asset.data.joint_names.index('a14_right_elbow')
    left_elbow_idx = asset.data.joint_names.index('a18_left_elbow')
    error += _negsqrd_exp(asset.data.joint_pos[:, right_elbow_idx])
    error += _negsqrd_exp(asset.data.joint_pos[:, left_elbow_idx])

    return error/4


# * --- Floating base rewards --- * #

def base_height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, base_height_target: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired base height
    base_height = asset.data.root_pos_w[:, 2]
    error = (base_height_target - base_height).flatten()
    return _negsqrd_exp(error)

def base_heading_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired base heading
    base_heading = asset.data.heading_w
    commands = env.command_manager.get_command(command_name)
    command_heading = torch.atan2(commands[:, 1], commands[:, 0])
    base_heading_error = torch.abs(math_utils.wrap_to_pi(command_heading - base_heading))
    return _neg_exp(base_heading_error, scale=torch.pi/2)

def base_z_orientation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking upright orientation
    error = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return _negsqrd_exp(error, scale=0.2)

def tracking_lin_vel_world_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired linear velocity command in the world frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, :2] - asset.data.root_lin_vel_w[:, :2]
    error *= 1./(1. + torch.abs(commands[:, :2]))
    return _negsqrd_exp(error, scale=1.).sum(dim=1)

def tracking_lin_vel_xy_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired linear velocity command in the base frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, :2] - asset.data.root_lin_vel_b[:, :2]
    error *= 1./(1. + torch.abs(commands[:, :2]))
    return _negsqrd_exp(error, scale=1.).sum(dim=1)

def tracking_yaw_vel_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired yaw velocity command
    commands = env.command_manager.get_command(command_name)
    error = commands[:, 2] - asset.data.root_ang_vel_b[:, 2]
    error *= 1./(1. + torch.abs(commands[:, 2]))
    return _negsqrd_exp(error, scale=1.)

def CAM_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(env.CM_bf[:, 3:5]), dim=1)

def dCAM_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Penalize the change in centroidal angular momentum in the xy plane
    return -torch.clamp_min(torch.sum(env.CM_bf[:, 3:5] * env.dCM_bf[:, 3:5], dim=1), 0.0)  # only positive build-up

def tracking_CAM_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired centroidal angular momentum
    error = env.CM_des[:, 5] - env.CM[:, 5]
    error *= 1./(1. + torch.abs(env.CM_des[:, 5]))
    return _negsqrd_exp(error, scale=0.5)

def CAM_compensation_reward(env: ManagerBasedRLEnv):
    """ Compensating the centroidal angular momentum generated by leg joint by arm joint """
    return _negsqrd_exp(env.CM_leg[:, -1] + env.CM_arm[:, -1])

# * --- Stepping rewards --- * #
def contact_schedule_reward(env: ManagerBasedRLEnv):
    """ Alternate right and left foot contacts
        First, right foot contacts (left foot swing), then left foot contacts (right foot swing) """ 
    return (env.foot_contact[:,0].int() - env.foot_contact[:,1].int()) * env.contact_schedule.squeeze(1)
    commands = env.command_manager.get_command("base_velocity")
    stand_mask = (torch.norm(commands[:,:2], dim=1) <= 0.2) & (torch.abs(commands[:,2]) <= 0.1)
    # return 3*(env.foot_contact[:,0].int() + env.foot_contact[:,1].int()) * stand_mask + \
    return (env.foot_contact[:,0].int() - env.foot_contact[:,1].int()) * env.contact_schedule.squeeze(1) * ~stand_mask

def contact_schedule_stepper_reward(env: ManagerBasedRLEnv):
    """ Alternate right and left foot contacts
        First, right foot contacts (left foot swing), then left foot contacts (right foot swing) 
        Contact should be made at the target foot position """ 
    contact_rewards = (env.foot_contact[:,0].int() - env.foot_contact[:,1].int()) * env.contact_schedule.squeeze(1)
    tracking_rewards = 3 * _neg_exp(env.step_location_offset[~env.foot_on_motion])
    return contact_rewards * tracking_rewards

# * --- Termination rewards --- * #
def termination_penalty(env: ManagerBasedRLEnv, group_name: str):
    # Penalize termination (except for time_outs)
    return -env.termination_manager.terminated[group_name].float()/env.step_dt


# * ######################### HELPER FUNCTIONS ############################## * #

REWARD_TRACKING_SIGMA = 0.25

def _neg_exp(x, scale=1):
    """ shorthand helper for negative exponential e^(-x/scale)
        scale: range of x
    """
    return torch.exp(-(x/scale)/REWARD_TRACKING_SIGMA)

def _negsqrd_exp(x, scale=1):
    """ shorthand helper for negative squared exponential e^(-(x/scale)^2)
        scale: range of x
    """
    return torch.exp(-torch.square(x/scale)/REWARD_TRACKING_SIGMA)