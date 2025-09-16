from __future__ import annotations
import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from isaaclab.envs import ManagerBasedEnv, VecEnvObs
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from ..utils import VanillaKeyboard, HumanoidVanillaRecorder

import isaaclab.utils.math as math_utils
import extensions.humanoid.utils as brl_utils

from extensions.humanoid.dynamics import PINOCCHIO_CASADI_FUNCTIONS_DIR
from casadi import Function
from cusadi import *

class HumanoidVanillaEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _init_buffers(self):
        super()._init_buffers()
        # * Initialize variables
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # * Observation variables
        self.contact_sensor = self.scene.sensors["contact_forces"]
        self.rf_contact_sensor = self.scene.sensors.get("rf_GRF", None)
        self.lf_contact_sensor = self.scene.sensors.get("lf_GRF", None)
        self.has_GRF_sensor = self.rf_contact_sensor is not None and self.lf_contact_sensor is not None
        self.body_feet_ids = [self.robot.data.body_names.index("right_foot"), self.robot.data.body_names.index("left_foot")]
        self.feet_ids = [self.contact_sensor.body_names.index('right_foot'), self.contact_sensor.body_names.index('left_foot')]
        self.phase_sin = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_cos = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_schedule = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_contact = torch.zeros(self.num_envs, len(self.feet_ids), dtype=torch.bool, device=self.device, requires_grad=False) # contacts on right & left sole
        self.foot_states = torch.zeros(self.num_envs, len(self.feet_ids), 7, dtype=torch.float, device=self.device, requires_grad=False) # num_envs x (left & right foot) x (x, y, z, quat)    
        self.foot_velocity = torch.zeros(self.num_envs, len(self.feet_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) # num_envs x (left & right foot) x (x, y, z)
        self.toe_heel_pos = torch.zeros(self.num_envs, 2 * len(self.feet_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) # num_envs x (right toe, heel / left toe, heel) x (x, y, z)
        self.foot_heading = torch.zeros(self.num_envs, len(self.feet_ids), dtype=torch.float, device=self.device, requires_grad=False) # num_envs x (left & right foot heading)  
        self.foot_height_vec = torch.tensor([0., 0., -0.04]).repeat(self.num_envs, 1).to(self.device)
        self.foot_to_toe_vec = torch.tensor([0.105, 0., 0.]).repeat(self.num_envs, 1).to(self.device)
        self.foot_to_heel_vec = torch.tensor([-0.045, 0., 0.]).repeat(self.num_envs, 1).to(self.device)
        self.vel_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # (v_x, v_y, w_z) wrt base frame

        self.rf_toe_GRF = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.rf_heel_GRF = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.lf_toe_GRF = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.lf_heel_GRF = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.rf_GRM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # Right foot's Ground Reaction Moment
        self.lf_GRM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # Left foot's Ground Reaction Moment

        self.step_period = self.cfg.commands.step_command.step_period * torch.ones(self.num_envs, 1, device=self.device, dtype=torch.long, requires_grad=False)
        self.full_step_period = 2 * self.step_period.clone()
        
        self.current_step = torch.zeros(self.num_envs, len(self.feet_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) # (left & right foot) x (x, y, heading) wrt base x,y-coordinate
        self.CoM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # Center of Mass
        
        self.leg_joints_name = ['a01_right_hip_yaw', 'a02_right_hip_abad', 'a03_right_hip_pitch', 'a04_right_knee', 'a05_right_ankle',
                                'a06_left_hip_yaw', 'a07_left_hip_abad', 'a08_left_hip_pitch', 'a09_left_knee', 'a10_left_ankle']
        self.arm_joints_name = ['a11_right_shoulder_pitch', 'a12_right_shoulder_abad', 'a13_right_shoulder_yaw', 'a14_right_elbow',
                                'a15_left_shoulder_pitch', 'a16_left_shoulder_abad', 'a17_left_shoulder_yaw', 'a18_left_elbow']

        self.leg_joints_idx = [self.robot.data.joint_names.index(joint) for joint in self.leg_joints_name]
        # self.arm_joints_idx = [self.robot.data.joint_names.index(joint) for joint in self.arm_joints_name]

        self.num_base_joints = 6 # Floating base joint
        self.num_leg_joints = len(self.leg_joints_name)
        self.num_arm_joints = len(self.arm_joints_name)
        self.num_total_joints = self.num_base_joints + self.num_leg_joints + self.num_arm_joints

        self.gen_coord_pin = torch.zeros(self.num_envs, 7 + self.num_leg_joints + self.num_arm_joints, device=self.device, requires_grad=False)
        self.gen_vel_body_pin = torch.zeros(self.num_envs, self.num_total_joints, device=self.device, requires_grad=False)
        self.gen_vel_body_pin_des = torch.zeros(self.num_envs, self.num_total_joints, device=self.device, requires_grad=False)
        self.gen_coord_pin_RS = torch.zeros(self.num_envs, 6 + self.num_leg_joints + self.num_arm_joints, device=self.device, requires_grad=False) # For RS, (roll, pitch, yaw) instead of (quaternion)

        self.CM = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Centroidal Momentum
        self.CM_base = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of base joint to Centroidal Momentum
        self.CM_leg = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of leg joints to Centroidal Momentum
        self.CM_arm = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of arm joints to Centroidal Momentum
        self.CM_des = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Desired Centroidal Momentum

        self.dCM = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Time derivative of Centroidal Momentum
        self.dCM_base = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of base joint to the time derivative of Centroidal Momentum
        self.dCM_leg = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of leg joints to the time derivative of Centroidal Momentum
        self.dCM_arm = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of arm joints to the time derivative of Centroidal Momentum
        self.dCM_des = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Desired time derivative of Centroidal Momentum

        self.rot_matrix_block = torch.zeros(self.num_envs, 6, 6, device=self.device, requires_grad=False)

        self.CM_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Centroidal Momentum in base frame
        self.CM_base_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # Contribution of base joint to Centroidal Momentum in base frame
        self.CM_leg_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.CM_arm_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.CM_des_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.dCM_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_base_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_leg_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_arm_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_des_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        self._set_CasADi_urdf_name()
        self.M_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M_{self.urdf_name}.casadi") # Mass matrix
        self.CMM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CMM_{self.urdf_name}.casadi") # Centroidal Momentum Matrix
        self.dCMM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCMM_{self.urdf_name}.casadi") # Time derivative of Centroidal Momentum Matrix
        self.CM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CM_{self.urdf_name}.casadi") # Centroidal Momentum
        self.dCM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCM_{self.urdf_name}.casadi") # Time derivative of Centroidal Momentum
        self.CoM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CoM_{self.urdf_name}.casadi") # Center of Mass
        
        self.base_pos_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_pos_{self.urdf_name}.casadi") # Base position
        self.base_rot_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_rot_{self.urdf_name}.casadi") # Base orientation in SO(3)

        self.cusadi_CMM_fn = CusadiFunction(self.CMM_fn, num_instances=self.num_envs, precision='float')
        self.cusadi_dCMM_fn = CusadiFunction(self.dCMM_fn, num_instances=self.num_envs, precision='float')
        self.cusadi_CM_fn = CusadiFunction(self.CM_fn, num_instances=self.num_envs, precision='float')
        self.cusadi_dCM_fn = CusadiFunction(self.dCM_fn, num_instances=self.num_envs, precision='float')
        self.cusadi_CoM_fn = CusadiFunction(self.CoM_fn, num_instances=self.num_envs, precision='float')
    
    def _set_CasADi_urdf_name(self):
        self.urdf_name = "humanoid_full_sf" # "humanoid_full_sf" or "humanoid_full_sf_400g"

    def _post_physics_step_callback(self):
        self.progress_within_step = self.episode_length_buf.unsqueeze(1) % self.full_step_period
        self.phase = self.progress_within_step / self.full_step_period
        
        self.phase_sin = torch.sin(2*torch.pi*self.phase)
        self.phase_cos = torch.cos(2*torch.pi*self.phase)

        self.contact_forces = self.contact_sensor.data.net_forces_w

        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        self.contact_schedule = self.smooth_sqr_wave(self.phase)
        
        self.vel_command = self.command_manager.get_command('base_velocity')

        self._compute_generalized_coordinates()

        self._calculate_GRF()

        self._calculate_CoM()

        self._calculate_centroidal_momentum()

        self._calculate_foot_velocity()

        pass

    def _compute_generalized_coordinates(self):
        """ Compute generalized coordinates and generalized velocities """
        gen_coord = torch.hstack((self.robot.data.root_link_pos_w,
                                  self.robot.data.root_link_quat_w,
                                  self.robot.data.joint_pos))
        gen_coord_pin = brl_utils.convert_gen_coord_from_isaaclab_to_pin(gen_coord)
        self.gen_coord_pin = torch.hstack((gen_coord_pin, torch.zeros(self.num_envs, self.num_arm_joints, device=self.device)))
        gen_vel_body = torch.hstack((self.robot.data.root_link_lin_vel_b,
                                     self.robot.data.root_link_ang_vel_b,
                                     self.robot.data.joint_vel))
        gen_vel_body_pin = brl_utils.convert_gen_vel_from_isaaclab_to_pin(gen_vel_body)
        self.gen_vel_body_pin = torch.hstack((gen_vel_body_pin, torch.zeros(self.num_envs, self.num_arm_joints, device=self.device)))
        gen_acc_body = torch.hstack((self.robot.data.root_link_lin_acc_b,
                                     self.robot.data.root_link_ang_acc_b,
                                     self.robot.data.joint_acc))
        gen_acc_body_pin = brl_utils.convert_gen_vel_from_isaaclab_to_pin(gen_acc_body)
        self.gen_acc_body_pin = torch.hstack((gen_acc_body_pin, torch.zeros(self.num_envs, self.num_arm_joints, device=self.device)))

        self.gen_vel_body_pin_des[:, 0] = self.vel_command[:, 0]
        self.gen_vel_body_pin_des[:, 1] = self.vel_command[:, 1]
        self.gen_vel_body_pin_des[:, 5] = self.vel_command[:, 2]

    def _calculate_GRF(self):
        """ Update foot contact forces """
        if not self.has_GRF_sensor:
            return
        
        # * Update foot states
        rfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[0], :7]
        lfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[1], :7]
        rfoot_height_vec_in_world = math_utils.quat_apply(rfoot_state[:,3:7], self.foot_height_vec)
        rfoot_to_toe_vec_in_world = math_utils.quat_apply(rfoot_state[:,3:7], self.foot_to_toe_vec)
        rfoot_to_heel_vec_in_world = math_utils.quat_apply(rfoot_state[:,3:7], self.foot_to_heel_vec)
        lfoot_height_vec_in_world = math_utils.quat_apply(lfoot_state[:,3:7], self.foot_height_vec)
        lfoot_to_toe_vec_in_world = math_utils.quat_apply(lfoot_state[:,3:7], self.foot_to_toe_vec)
        lfoot_to_heel_vec_in_world = math_utils.quat_apply(lfoot_state[:,3:7], self.foot_to_heel_vec)

        self.foot_states[:,0,:3] = rfoot_state[:,:3] + rfoot_height_vec_in_world
        self.foot_states[:,0,3:7] = rfoot_state[:,3:7]
        self.foot_states[:,1,:3] = lfoot_state[:,:3] + lfoot_height_vec_in_world
        self.foot_states[:,1,3:7] = lfoot_state[:,3:7]
        self.toe_heel_pos[:,0,:] = self.foot_states[:,0,:3] + rfoot_to_toe_vec_in_world
        self.toe_heel_pos[:,1,:] = self.foot_states[:,0,:3] + rfoot_to_heel_vec_in_world
        self.toe_heel_pos[:,2,:] = self.foot_states[:,1,:3] + lfoot_to_toe_vec_in_world
        self.toe_heel_pos[:,3,:] = self.foot_states[:,1,:3] + lfoot_to_heel_vec_in_world

        # * Calculate GRF
        tol = 1e-2

        rel_idx = torch.arange(self.rf_contact_sensor.cfg.max_contact_data_count_per_env, device=self.device).unsqueeze(0)  # Shape: (1, max_contacts)
        rf_valid_mask = rel_idx < self.rf_contact_sensor.data.GRF_count_buffer.unsqueeze(1)
        lf_valid_mask = rel_idx < self.lf_contact_sensor.data.GRF_count_buffer.unsqueeze(1)

        rf_toe_diff = (self.toe_heel_pos[:,0].unsqueeze(1) - self.rf_contact_sensor.data.GRF_points_buffer)
        rf_toe_dist = rf_toe_diff.norm(dim=-1)
        rf_toe_mask = (rf_toe_dist < tol) * rf_valid_mask
        rf_heel_diff = (self.toe_heel_pos[:,1].unsqueeze(1) - self.rf_contact_sensor.data.GRF_points_buffer)
        rf_heel_dist = rf_heel_diff.norm(dim=-1)
        rf_heel_mask = (rf_heel_dist < tol) * rf_valid_mask
        lf_toe_diff = (self.toe_heel_pos[:,2].unsqueeze(1) - self.lf_contact_sensor.data.GRF_points_buffer)
        lf_toe_dist = lf_toe_diff.norm(dim=-1)
        lf_toe_mask = (lf_toe_dist < tol) * lf_valid_mask
        lf_heel_diff = (self.toe_heel_pos[:,3].unsqueeze(1) - self.lf_contact_sensor.data.GRF_points_buffer)
        lf_heel_dist = lf_heel_diff.norm(dim=-1)
        lf_heel_mask = (lf_heel_dist < tol) * lf_valid_mask

        self.rf_toe_GRF = (self.rf_contact_sensor.data.GRF_forces_buffer * rf_toe_mask.unsqueeze(2)).sum(dim=1)
        self.rf_heel_GRF = (self.rf_contact_sensor.data.GRF_forces_buffer * rf_heel_mask.unsqueeze(2)).sum(dim=1)
        self.lf_toe_GRF = (self.lf_contact_sensor.data.GRF_forces_buffer * lf_toe_mask.unsqueeze(2)).sum(dim=1)
        self.lf_heel_GRF = (self.lf_contact_sensor.data.GRF_forces_buffer * lf_heel_mask.unsqueeze(2)).sum(dim=1)

        # * Calculate the Ground Reaction Moment (GRM) = (p_toe - p_foot) x GRF_toe + (p_heel - p_foot) x GRF_heel
        self.rf_GRM = torch.linalg.cross(rfoot_to_toe_vec_in_world, self.rf_toe_GRF) + torch.linalg.cross(rfoot_to_heel_vec_in_world, self.rf_heel_GRF)
        self.lf_GRM = torch.linalg.cross(lfoot_to_toe_vec_in_world, self.lf_toe_GRF) + torch.linalg.cross(lfoot_to_heel_vec_in_world, self.lf_heel_GRF)

    def _calculate_CoM(self):
        """ Calculate the CoM position of the robot """
        self.cusadi_CoM_fn.evaluate([self.gen_coord_pin.to('cuda')])
        self.CoM = self.cusadi_CoM_fn.outputs_sparse[0].to(self.device).float()

    def _calculate_centroidal_momentum(self):
        """ Calculate the centroidal momelntum of the robot
            IsaacLab's mass matrix is projected to the root link's CoM frame
            We need to convert it to whole body's CoM frame
            Convert from root_link_CoM (world frame) -> root_link (base frame) -> CoM (world frame)
        """
        self.cusadi_CMM_fn.evaluate([self.gen_coord_pin.to('cuda')])
        CMM = self.cusadi_CMM_fn.outputs_sparse[0].reshape(self.num_envs, -1, self.num_base_joints).permute(0,2,1).to(self.device)
        self.cusadi_dCMM_fn.evaluate([self.gen_coord_pin.to('cuda'), self.gen_vel_body_pin.to('cuda')])
        dCMM = self.cusadi_dCMM_fn.outputs_sparse[0].reshape(self.num_envs, -1, self.num_base_joints).permute(0,2,1).to(self.device)

        self.CM = (CMM @ self.gen_vel_body_pin.unsqueeze(2)).squeeze(2).float()
        self.CM_base = (CMM[:,:,:self.num_base_joints] @ self.gen_vel_body_pin[:,:self.num_base_joints].unsqueeze(2)).squeeze(2).float() # A(q) * qdot
        self.CM_leg = (CMM[:,:,self.num_base_joints:self.num_base_joints+self.num_leg_joints] @ self.gen_vel_body_pin[:,self.num_base_joints:self.num_base_joints+self.num_leg_joints].unsqueeze(2)).squeeze(2).float() # A(q) * qdot
        self.CM_arm = (CMM[:,:,self.num_base_joints+self.num_leg_joints:] @ self.gen_vel_body_pin[:,self.num_base_joints+self.num_leg_joints:].unsqueeze(2)).squeeze(2).float() # A(q) * qdot
        self.CM_des = (CMM @ self.gen_vel_body_pin_des.unsqueeze(2)).squeeze(2).float()

        self.dCM = (dCMM @ self.gen_vel_body_pin.unsqueeze(2)).squeeze(2).float() + (CMM @ self.gen_acc_body_pin.unsqueeze(2)).squeeze(2).float() # dCM = dA(q) * qdot + A(q) * qddot
        self.dCM_base = (dCMM[:,:,:self.num_base_joints] @ self.gen_vel_body_pin[:,:self.num_base_joints].unsqueeze(2)).squeeze(2).float() + \
                         (CMM[:,:,:self.num_base_joints] @ self.gen_acc_body_pin[:,:self.num_base_joints].unsqueeze(2)).squeeze(2).float()
        self.dCM_leg = (dCMM[:,:,self.num_base_joints:self.num_base_joints+self.num_leg_joints] @ self.gen_vel_body_pin[:,self.num_base_joints:self.num_base_joints+self.num_leg_joints].unsqueeze(2)).squeeze(2).float() + \
                        (CMM[:,:,self.num_base_joints:self.num_base_joints+self.num_leg_joints] @ self.gen_acc_body_pin[:,self.num_base_joints:self.num_base_joints+self.num_leg_joints].unsqueeze(2)).squeeze(2).float()
        self.dCM_arm = (dCMM[:,:,self.num_base_joints+self.num_leg_joints:] @ self.gen_vel_body_pin[:,self.num_base_joints+self.num_leg_joints:].unsqueeze(2)).squeeze(2).float() + \
                        (CMM[:,:,self.num_base_joints+self.num_leg_joints:] @ self.gen_acc_body_pin[:,self.num_base_joints+self.num_leg_joints:].unsqueeze(2)).squeeze(2).float()

        rot_matrix = math_utils.matrix_from_quat(self.robot.data.root_link_quat_w).permute(0,2,1) # Rotation matrix from world frame to base frame
        self.rot_matrix_block[:,:3,:3] = rot_matrix
        self.rot_matrix_block[:,3:,3:] = rot_matrix

        self.CM_bf = (self.rot_matrix_block @ self.CM.unsqueeze(2)).squeeze(2)
        self.CM_base_bf = (self.rot_matrix_block @ self.CM_base.unsqueeze(2)).squeeze(2)
        self.CM_leg_bf = (self.rot_matrix_block @ self.CM_leg.unsqueeze(2)).squeeze(2)
        self.CM_arm_bf = (self.rot_matrix_block @ self.CM_arm.unsqueeze(2)).squeeze(2)
        self.CM_des_bf = (self.rot_matrix_block @ self.CM_des.unsqueeze(2)).squeeze(2)

        self.dCM_bf = (self.rot_matrix_block @ self.dCM.unsqueeze(2)).squeeze(2)
        self.dCM_base_bf = (self.rot_matrix_block @ self.dCM_base.unsqueeze(2)).squeeze(2)
        self.dCM_leg_bf = (self.rot_matrix_block @ self.dCM_leg.unsqueeze(2)).squeeze(2)
        self.dCM_arm_bf = (self.rot_matrix_block @ self.dCM_arm.unsqueeze(2)).squeeze(2)

    def _calculate_foot_velocity(self):
        """ Calculate the foot velocity in world frame """
        self.foot_velocity[:,0] = self.robot.data.body_state_w[:, self.body_feet_ids[0], 7:10]
        self.foot_velocity[:,1] = self.robot.data.body_state_w[:, self.body_feet_ids[1], 7:10]

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.phase[env_ids] = 0
        self.phase_sin[env_ids] = torch.sin(2*torch.pi*self.phase[env_ids])
        self.phase_cos[env_ids] = torch.cos(2*torch.pi*self.phase[env_ids])
        self.contact_schedule[env_ids] = self.smooth_sqr_wave(self.phase[env_ids])

        self.rf_toe_GRF[env_ids] = 0.
        self.rf_heel_GRF[env_ids] = 0.
        self.lf_toe_GRF[env_ids] = 0.
        self.lf_heel_GRF[env_ids] = 0.

    def _setup_keyboard_interface(self):
        self.keyboard_interface = VanillaKeyboard(self)

    def _setup_recorder(self):
        self.recorder = HumanoidVanillaRecorder(fps=int(1/self.step_dt), total_weight=self.total_weight[0], robot_cfg=self.robot.cfg)

    def _get_log_data(self):
        robot_idx = 0
        log_data = {
            'root_lin_vel_b': self.robot.data.root_lin_vel_b[robot_idx],
            'root_ang_vel_b': self.robot.data.root_ang_vel_b[robot_idx],
            'vel_command': self.vel_command[robot_idx],
            'CM': self.CM[robot_idx],
            'CM_base': self.CM_base[robot_idx],
            'CM_leg': self.CM_leg[robot_idx],
            'CM_arm': self.CM_arm[robot_idx],
            'CM_des': self.CM_des[robot_idx],
            'CM_bf': self.CM_bf[robot_idx],
            'CM_base_bf': self.CM_base_bf[robot_idx],
            'CM_leg_bf': self.CM_leg_bf[robot_idx],
            'CM_arm_bf': self.CM_arm_bf[robot_idx],
            'CM_des_bf': self.CM_des_bf[robot_idx],
            'dCM': self.dCM[robot_idx],
            'dCM_base': self.dCM_base[robot_idx],
            'dCM_leg': self.dCM_leg[robot_idx],
            'dCM_arm': self.dCM_arm[robot_idx],
            'dCM_des': self.dCM_des[robot_idx],
            'dCM_bf': self.dCM_bf[robot_idx],
            'dCM_base_bf': self.dCM_base_bf[robot_idx],
            'dCM_leg_bf': self.dCM_leg_bf[robot_idx],
            'dCM_arm_bf': self.dCM_arm_bf[robot_idx],
            'dCM_des_bf': self.dCM_des_bf[robot_idx],
            'rf_toe_GRF': self.rf_toe_GRF[robot_idx],
            'rf_heel_GRF': self.rf_heel_GRF[robot_idx],
            'lf_toe_GRF': self.lf_toe_GRF[robot_idx],
            'lf_heel_GRF': self.lf_heel_GRF[robot_idx],
            'rf_GRM': self.rf_GRM[robot_idx],
            'lf_GRM': self.lf_GRM[robot_idx],
            'leg_joint_pos': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.joint_pos)[robot_idx, :self.num_leg_joints],
            'leg_joint_pos_target': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.default_joint_pos + self.action_manager.action['joint_pos'])[robot_idx, :self.num_leg_joints],
            'leg_joint_vel': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.joint_vel)[robot_idx, :self.num_leg_joints],
            'leg_joint_torque': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.applied_torque)[robot_idx, :self.num_leg_joints],
            'phase': self.phase[robot_idx],
            'foot_contact': self.foot_contact[robot_idx],
            'contact_schedule': self.contact_schedule[robot_idx],
            'foot_velocity': self.foot_velocity[robot_idx],
        }
        return log_data


    # ##################### HELPER FUNCTIONS ################################## #

    def smooth_sqr_wave(self, phase, eps=1.0):
        p = 2.*torch.pi*phase
        return torch.sin(p) / torch.sqrt(torch.sin(p)**2. + eps**2.)
    
    def smooth_sqr_wave_advanced(self, phase, eps=0.2, a=0.2):
        """ Advanced contact scheduler containing the flying phase
            a: fraction of the flying phase 
        """
        t = (1 - a) / 2

        # Piecewise function for the first half of the cycle (right foot)
        r = 2.*torch.pi*phase / (1 - a)
        r_output = torch.sin(r) / torch.sqrt(torch.sin(r)**2 + eps**2)

        # Piecewise function for the second half of the cycle (left foot)
        l = 2.*torch.pi*(phase - (0.5 - t)) / (1 - a)
        l_output = torch.sin(l) / torch.sqrt(torch.sin(l)**2 + eps**2)

        # Create masks
        mask_r = (phase < t)                 # First part of gait cycle (right foot)
        mask_l = (0.5 <= phase) & (phase < 0.5 + t)    # Second part of gait cycle (left foot)

        # Apply masks to calculate the output
        output = r_output * mask_r + l_output * mask_l

        return output
    
""" Code Explanation 
1. 
['base', 'right_hip_yaw', 'right_hip_abad', 'right_upper_leg', 'right_lower_leg', 'right_foot', 
         'right_shoulder', 'right_shoulder_2', 'right_upper_arm', 'right_lower_arm', 'right_hand',
         'left_hip_yaw', 'left_hip_abad', 'left_upper_leg', 'left_lower_leg', 'left_foot', 
         'left_shoulder', 'left_shoulder_2', 'left_upper_arm', 'left_lower_arm', 'left_hand']

2.
['a_1_right_hip_yaw', 'a_1_right_shoulder_pitch', 'a_5_left_shoulder_pitch', 'a_6_left_hip_yaw', 
'a_2_right_hip_abad', 'a_2_right_shoulder_abad', 'a_6_left_shoulder_abad', 'a_7_left_hip_abad', 
'a_3_right_hip_pitch', 'a_3_right_shoulder_yaw', 'a_7_left_shoulder_yaw', 'a_8_left_hip_pitch',
'a_4_right_knee', 'a_4_right_elbow', 'a_8_left_elbow', 'a_9_left_knee', 'a_5_right_ankle', 'a_0_left_ankle']

3. About "root_physx_view"
self.robot = self.scene["robot"]

tau = M(q) * qddot + C(q, qdot) * qdot + G(q)
M(q) : self.robot.root_physx_view.get_mass_matrices()
C(q, qdot) : self.robot.root_physx_view.get_coriolis_and_centrifugal_forces()
G(q) : self.robot.root_physx_view.get_generalized_gravity_forces()

jacobian : self.robot.root_physx_view.get_jacobians()

4. Contact forces
It seems when you use 'cpu', 
you should set 'self.disable_contact_processing = False' in the SimulationCfg to access the contact forces.

However, when you use 'gpu', 
you can access the contact forces even with setting 'self.disable_contact_processing = True'.
If you set `self.disable_contact_processing = False`, it will slow down the training speed.

5.
self.env.render()

def _calculate_centroidal_momentum(self):    
    gen_coord = torch.hstack((self.robot.data.root_pos_w,
                                self.robot.data.root_quat_w,
                                self.robot.data.joint_pos))
    gen_coord_pin = brl_utils.convert_gen_coord_from_isaaclab_to_pin(gen_coord)
    gen_coord_pin = torch.hstack((gen_coord_pin, torch.zeros(self.num_envs, self.num_arm_joints, device=self.device)))
    gen_vel_body = torch.hstack((self.robot.data.root_lin_vel_b,
                                    self.robot.data.root_ang_vel_b,
                                    self.robot.data.joint_vel))
    gen_vel_body_pin = brl_utils.convert_gen_vel_from_isaaclab_to_pin(gen_vel_body)
    gen_vel_body_pin = torch.hstack((gen_vel_body_pin, torch.zeros(self.num_envs, self.num_arm_joints, device=self.device)))
        
    mass_matrices_lab = self.robot.root_physx_view.get_generalized_mass_matrices() # Projected to root_com frame
    body_rot_matrix = math_utils.matrix_from_quat(self.robot.data.root_quat_w)
    rootCoM_to_root = self.robot.data.root_pos_w - self.robot.data.root_com_pos_w
    adjoint_matrix_twist = brl_utils.adjoint_matrix_twist(body_rot_matrix, rootCoM_to_root)
    block_adjoint_matrix_twist = torch.zeros(self.num_envs, 24, 24, device=self.device)
    block_adjoint_matrix_twist[:, :6, :6] = adjoint_matrix_twist
    block_adjoint_matrix_twist[:, 6:, 6:] = torch.eye(18).expand(self.num_envs, 18, 18)
    mass_matrices_base = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_lab @ block_adjoint_matrix_twist # Projected to base frame / generalized velocity in base frame

    root_to_CoM = self.CoM - self.robot.data.root_pos_w
    adjoint_matrix_twist = brl_utils.adjoint_matrix_twist(body_rot_matrix.permute(0,2,1), root_to_CoM)
    block_adjoint_matrix_twist = torch.zeros(self.num_envs, 24, 24, device=self.device)
    block_adjoint_matrix_twist[:, :6, :6] = adjoint_matrix_twist
    block_adjoint_matrix_twist[:, 6:, 6:] = torch.eye(18).expand(self.num_envs, 18, 18)
    # composite_rigid_body_inertia = adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_base[:,:6,:6] @ adjoint_matrix_twist # Projected to CoM frame / generalized velocity in base frame
    mass_matrices_CMM = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_base # Projected to CoM frame / generalized velocity in base frame
    # self.CM = (mass_matrices_CMM[:,:6,:] @ gen_vel_body.unsqueeze(2)).squeeze(2) # A(q) * qdot

def _calculate_CoM(self):
    body_masses = self.robot.root_physx_view.get_masses().to(self.device)
    self.CoM = (self.robot.data.body_com_pos_w * body_masses.unsqueeze(2)).sum(dim=1) / body_masses.sum(dim=1, keepdim=True)
    self.CoM = (self.robot.data.body_com_pos_w * self.body_masses).sum(dim=1) * self.inv_total_mass

    
! Be careful. There's 3 types of link representation
1. root_state_w : pos (origin) / quat (origin) / lin_vel (CoM) / ang_vel (CoM)
2. root_link_state_w : pos (origin) / quat (origin) / lin_vel (origin) / ang_vel (origin)
3. root_com_state_w_ : pos (CoM) / quat (origin) / lin_vel (CoM) / ang_vel (CoM)
"""
