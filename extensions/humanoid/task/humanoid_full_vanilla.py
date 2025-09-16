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
from ..utils import VanillaKeyboard, HumanoidFullVanillaRecorder

import isaaclab.utils.math as math_utils
import extensions.humanoid.utils as brl_utils
from extensions.humanoid.task import HumanoidVanillaEnv

class HumanoidFullVanillaEnv(HumanoidVanillaEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _init_buffers(self):
        super()._init_buffers()
        self.arm_joints_idx = [self.robot.data.joint_names.index(joint) for joint in self.arm_joints_name]

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

        pass

    def _compute_generalized_coordinates(self):
        """ Compute generalized coordinates and generalized velocities """
        gen_coord = torch.hstack((self.robot.data.root_link_pos_w,
                                  self.robot.data.root_link_quat_w,
                                  self.robot.data.joint_pos))
        self.gen_coord_pin = brl_utils.convert_gen_coord_from_isaaclab_to_pin(gen_coord)
        gen_vel_body = torch.hstack((self.robot.data.root_link_lin_vel_b,
                                     self.robot.data.root_link_ang_vel_b,
                                     self.robot.data.joint_vel))
        self.gen_vel_body_pin = brl_utils.convert_gen_vel_from_isaaclab_to_pin(gen_vel_body)
        gen_acc_body = torch.hstack((self.robot.data.root_link_lin_acc_b,
                                     self.robot.data.root_link_ang_acc_b,
                                     self.robot.data.joint_acc))
        self.gen_acc_body_pin = brl_utils.convert_gen_vel_from_isaaclab_to_pin(gen_acc_body)

        self.gen_vel_body_pin_des[:, 0] = self.vel_command[:, 0]
        self.gen_vel_body_pin_des[:, 1] = self.vel_command[:, 1]
        self.gen_vel_body_pin_des[:, 5] = self.vel_command[:, 2]

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def _setup_keyboard_interface(self):
        self.keyboard_interface = VanillaKeyboard(self)

    def _setup_recorder(self):
        self.recorder = HumanoidFullVanillaRecorder(fps=int(1/self.step_dt), total_weight=self.total_weight[0], robot_cfg=self.robot.cfg)

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
            'arm_joint_pos': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.joint_pos)[robot_idx, self.num_leg_joints:],
            'arm_joint_pos_target': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.default_joint_pos + self.action_manager.action['joint_pos'])[robot_idx, self.num_leg_joints:],
            'arm_joint_vel': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.joint_vel)[robot_idx, self.num_leg_joints:],
            'arm_joint_torque': brl_utils.convert_joint_order_from_isaaclab_to_pin(self.robot.data.applied_torque)[robot_idx, self.num_leg_joints:],
            'phase': self.phase[robot_idx],
            'foot_contact': self.foot_contact[robot_idx],
        }
        return log_data


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

6. Match mass_matrix from pinocchio and isaaclab

from extensions.humanoid.dynamics import PINOCCHIO_CASADI_FUNCTIONS_DIR
from casadi import Function

self.M_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M.casadi") # Mass matrix
self.C_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/C.casadi") # Coriolis and Centrifugal forces vector
self.G_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/G.casadi") # Gravity forces vector

gen_vel_world = torch.hstack((self.robot.data.root_link_lin_vel_w,
                                self.robot.data.root_link_ang_vel_w,
                                self.robot.data.joint_vel)).unsqueeze(2)
CMM = self.robot.root_physx_view.get_generalized_mass_matrices()[:,:6,:]
rootCoM_to_CoM = self.CoM - self.robot.data.root_com_pos_w  # Centroidal Momentum Matrix (CMM) is (6, 6+nj) projected in CoM frame aligned with world frame
adjoint_matrix_twist = brl_utils.adjoint_matrix_twist(torch.eye(3).repeat(self.num_envs,1,1), rootCoM_to_CoM)
block_adjoint_matrix_twist = torch.zeros(self.num_envs, 24, 24)
block_adjoint_matrix_twist[:, :6, :6] = adjoint_matrix_twist
block_adjoint_matrix_twist[:, 6:, 6:] = torch.eye(18).expand(self.num_envs, 18, 18)
CoM_mass_matrix = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_lab
CMM = CoM_mass_matrix[:,:6,:] # A(q)
CAM = (CMM @ gen_vel_world).squeeze(2) # A(q) * qdot

gen_coord = torch.hstack((self.robot.data.root_pos_w,
                            self.robot.data.root_quat_w,
                            self.robot.data.joint_pos))
gen_coord = brl_utils.convert_gen_coord_from_isaaclab_to_pin(gen_coord).cpu().numpy()
gen_vel_body = torch.hstack((self.robot.data.root_link_lin_vel_b,
                                self.robot.data.root_link_ang_vel_b,
                                self.robot.data.joint_vel)).unsqueeze(2)
# gen_vel_body = brl_utils.convert_gen_vel_from_isaaclab_to_pin(gen_vel_body)
mass_matrices_pin = torch.tensor(np.array(self.M_fn(gen_coord)), device=self.device, dtype=torch.float32)
mass_matrices_pin = brl_utils.convert_mass_matrix_from_pin_to_isaaclab(mass_matrices_pin)

mass_matrices_lab = self.robot.root_physx_view.get_generalized_mass_matrices()
body_rot_matrix = math_utils.matrix_from_quat(self.robot.data.root_quat_w)
root_to_rootCoM = self.robot.data.root_com_pos_w - self.robot.data.root_pos_w
adjoint_matrix_twist = brl_utils.adjoint_matrix_twist(body_rot_matrix, -root_to_rootCoM)
adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_lab[:,:6,:6] @ adjoint_matrix_twist
block_adjoint_matrix_twist = torch.block_diag(adjoint_matrix_twist.squeeze(0), torch.eye(18,18)).unsqueeze(0)
mass_matrices_test = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_lab @ block_adjoint_matrix_twist

rootCoM_to_CoM = self.CoM - self.robot.data.root_com_pos_w
adjoint_matrix_twist = brl_utils.adjoint_matrix_twist(torch.eye(3), rootCoM_to_CoM)
block_adjoint_matrix_twist = torch.block_diag(adjoint_matrix_twist.squeeze(0), torch.eye(18,18)).unsqueeze(0)
CMM2 = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_lab @ block_adjoint_matrix_twist

root_to_CoM = self.CoM - self.robot.data.root_pos_w
adjoint_matrix_twist = brl_utils.adjoint_matrix_twist(body_rot_matrix.permute(0,2,1), root_to_CoM)
block_adjoint_matrix_twist = torch.block_diag(adjoint_matrix_twist.squeeze(0), torch.eye(18,18)).unsqueeze(0)
pin_CMM = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_pin
pin_CAM = (pin_CMM[:,:6,:] @ gen_vel_body).squeeze(2)
test_CMM = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices_test
test_CAM = (test_CMM[:,:6,:] @ gen_vel_body).squeeze(2)

adjoint_matrix_twist_1_to_com = brl_utils.adjoint_matrix_twist(body_rot_matrix, -root_to_rootCoM)
block_adjoint_matrix_twist = torch.zeros(self.num_envs, 24, 24)
block_adjoint_matrix_twist[:, :6, :6] = adjoint_matrix_twist_1_to_com
block_adjoint_matrix_twist[:, 6:, 6:] = torch.eye(18).expand(self.num_envs, 18, 18)
gen_vel_com_world = block_adjoint_matrix_twist @ gen_vel_body

print('Mass matrix difference: ', (mass_matrices_test - mass_matrices_pin).max())

print('Energy PIN: ', gen_vel_body.permute(0,2,1) @ mass_matrices_pin @ gen_vel_body)
print('Energy TEST: ', gen_vel_body.permute(0,2,1) @ mass_matrices_test @ gen_vel_body)
print('Energy BASE: ', gen_vel_body.permute(0,2,1) @ mass_matrices_base @ gen_vel_body)
print('Energy LAB: ', gen_vel_com_world.permute(0,2,1) @ mass_matrices_lab @ gen_vel_com_world)
pass


"""
