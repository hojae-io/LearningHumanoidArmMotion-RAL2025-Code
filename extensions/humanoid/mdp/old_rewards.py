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

# * --- Regularization Rewards --- * #

def action_smoothness1(env: ManagerBasedRLEnv):
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    dt2 = (env.step_dt*env.cfg.decimation)**2
    error = torch.square(action - prev_action)/dt2
    return -torch.sum(error, dim=1)

def action_smoothness2(env: ManagerBasedRLEnv):
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev_action = env.action_manager.prev_prev_action
    dt2 = (env.step_dt*env.cfg.decimation)**2
    error = torch.square(action - 2*prev_action + prev_prev_action)/dt2
    return -torch.sum(error, dim=1)

def joint_torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.applied_torque), dim=1)

def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.joint_vel), dim=1)

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
    return -torch.sum(out_of_limits, dim=1)

def joint_torque_limits_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize joint torques too close to the limit
    return -torch.sum((torch.abs(asset.data.applied_torque) - asset.data.soft_joint_torque_limits).clip(min=0.), dim=1)

def joint_regularization_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Reward joint poses and symmetry
    error = 0.

    # Yaw joints regularization around 0
    right_hip_yaw_idx = asset.data.joint_names.index('a01_right_hip_yaw')
    left_hip_yaw_idx = asset.data.joint_names.index('a06_left_hip_yaw')
    error += _negsqrd_exp((asset.data.joint_pos[:, right_hip_yaw_idx]))
    error += _negsqrd_exp((asset.data.joint_pos[:, left_hip_yaw_idx]))
        
    # Ab/ad joint symmetry
    # error += self._negsqrd_exp(
    #     (self.dof_pos[:, 1] - self.dof_pos[:, 6])
    #     / self.scales['dof_pos'])
    right_hip_abad_idx = asset.data.joint_names.index('a02_right_hip_abad')
    left_hip_abad_idx = asset.data.joint_names.index('a07_left_hip_abad')
    error += _negsqrd_exp((asset.data.joint_pos[:, right_hip_abad_idx]))
    error += _negsqrd_exp((asset.data.joint_pos[:, left_hip_abad_idx]))

    # Pitch joint symmetry
    # error += self._negsqrd_exp(
    #     (self.dof_pos[:, 2] + self.dof_pos[:, 7])
    #     / self.scales['dof_pos'])
    # error += self._negsqrd_exp((self.dof_pos[:, 3] - self.dof_pos[:, 8]) / self.scales['dof_pos']) # knee

    # # Shoulder pitch symmetry
    # right_shoulder_pitch_idx = asset.data.joint_names.index('a11_right_shoulder_pitch')
    # left_shoulder_pitch_idx = asset.data.joint_names.index('a15_left_shoulder_pitch')
    # error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_pitch_idx] + 
    #                       asset.data.joint_pos[:, left_shoulder_pitch_idx])
    # # error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_pitch_idx]) * 0.3
    # # error += _negsqrd_exp(asset.data.joint_pos[:, left_shoulder_pitch_idx]) * 0.3
    
    # # Shoulder ab/ad regularization around 0
    # right_shoulder_abad_idx = asset.data.joint_names.index('a12_right_shoulder_abad')
    # left_shoulder_abad_idx = asset.data.joint_names.index('a16_left_shoulder_abad')
    # error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_abad_idx]) * 0.2
    # error += _negsqrd_exp(asset.data.joint_pos[:, left_shoulder_abad_idx]) * 0.2

    # # Shoulder yaw regularization around 0
    # right_shoulder_yaw_idx = asset.data.joint_names.index('a13_right_shoulder_yaw')
    # left_shoulder_yaw_idx = asset.data.joint_names.index('a17_left_shoulder_yaw')
    # error += _negsqrd_exp(asset.data.joint_pos[:, right_shoulder_yaw_idx]) * 0.2
    # error += _negsqrd_exp(asset.data.joint_pos[:, left_shoulder_yaw_idx]) * 0.2

    # # Elbow regularization around 0
    # right_elbow_idx = asset.data.joint_names.index('a14_right_elbow')
    # left_elbow_idx = asset.data.joint_names.index('a18_left_elbow')
    # error += _negsqrd_exp(asset.data.joint_pos[:, right_elbow_idx]) * 0.2
    # error += _negsqrd_exp(asset.data.joint_pos[:, left_elbow_idx]) * 0.2

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
    base_heading_error = torch.abs(wrap_to_pi(command_heading - base_heading))
    return _neg_exp(base_heading_error, a=torch.pi/2)

def base_z_orientation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking upright orientation
    error = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return _negsqrd_exp(error, a=0.2)

def tracking_lin_vel_world_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired linear velocity command in the world frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, :2] - asset.data.root_lin_vel_w[:, :2]
    error *= 1./(1. + torch.abs(commands[:, :2]))
    return _negsqrd_exp(error, a=1.).sum(dim=1)

# * --- Stepping rewards --- * #
def contact_schedule_reward(env: ManagerBasedRLEnv):
    """ Alternate right and left foot contacts
        First, right foot contacts (left foot swing), then left foot contacts (right foot swing) """ 
    return (env.foot_contact[:,0].int() - env.foot_contact[:,1].int()) * env.contact_schedule.squeeze(1)

# * --- Termination rewards --- * #
def termination_penalty(env: ManagerBasedRLEnv):
    # Penalize termination (except for time_outs)
    return -env.termination_manager.terminated.float()/env.step_dt



""" Original """
# --- Task Rewards --- #

def air_time_reward(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    command = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1).unsqueeze(-1).expand(-1, 2)
    mode_time = 0.5

    max_time = torch.max(contact_time, air_time)
    min_time = torch.clamp(max_time, max=mode_time)
    stance_reward = torch.clamp(contact_time-air_time, max=mode_time, min=-mode_time)
    stepping_reward = torch.where(max_time < mode_time, min_time, 0.0)
    
    reward = torch.where(command > 0.0, stepping_reward, stance_reward)
    return torch.sum(reward, dim=1)

def foot_clearance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_diff = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    foot_vel_tanh = torch.tanh(tanh_mult*torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.square(foot_z_diff) * foot_vel_tanh
    return torch.exp(-torch.sum(reward, dim=1)/std)

def gait_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float):
    pass

def track_ang_vel_z_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error = torch.square(commands-asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error/std)

def track_lin_vel_xy_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.linalg.norm(commands-asset.data.root_lin_vel_b[:, :2], dim=1)
    return torch.exp(-lin_vel_error/std)

# --- Regularization Penalties --- #

def action_smoothness_first_order_penalty(env: ManagerBasedRLEnv):
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.linalg.norm((action-prev_action), dim=1)

# TODO: create new action_manager to support prev_prev_action
def action_smoothness_second_order_penalty(env: ManagerBasedRLEnv):
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev_action = env.action_manager.prev_prev_action
    return torch.linalg.norm((action-2.0*prev_action+prev_prev_action), dim=1)

def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    mode_time = 0.5
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_air_time_clamped = torch.clamp(last_air_time, max=mode_time)
    last_contact_time_clamped = torch.clamp(last_contact_time, max=mode_time)
    return torch.var(last_air_time_clamped, dim=1) + torch.var(last_contact_time_clamped, dim=1)

def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    base_lin_vel_z = asset.data.root_lin_vel_b[:, 2]
    base_ang_vel_rp = asset.data.root_ang_vel_b[:, :2]
    return 0.8*torch.square(base_lin_vel_z)+0.2*torch.sum(torch.abs(base_ang_vel_rp), dim=1)

def base_orentation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)

def foot_slip_penalty(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    
    reward = is_contact*foot_planar_velocity
    return torch.sum(reward, dim=1)

def joint_position_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_pos-asset.data.default_joint_pos), dim=1)

# def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.linalg.norm((asset.data.joint_vel), dim=1)

def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)

# def joint_torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.linalg.norm((asset.data.applied_torque), dim=1)


# * ######################### HELPER FUNCTIONS ############################## * #

REWARD_TRACKING_SIGMA = 0.25

def _neg_exp(x, a=1):
    """ shorthand helper for negative exponential e^(-x/a)
        a: range of x
    """
    return torch.exp(-(x/a)/REWARD_TRACKING_SIGMA)

def _negsqrd_exp(x, a=1):
    """ shorthand helper for negative squared exponential e^(-(x/a)^2)
        a: range of x
    """
    return torch.exp(-torch.square(x/a)/REWARD_TRACKING_SIGMA)