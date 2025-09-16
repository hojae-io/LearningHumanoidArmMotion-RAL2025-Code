"""Recorder interface for each environment."""

from isaaclab.managers import RecorderBase, VideoRecorder

import os
from extensions import ISAACLAB_BRL_ROOT_DIR
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from PIL import Image
from collections import defaultdict
from typing import Callable, Dict
from isaaclab.assets.articulation import ArticulationCfg
import seaborn as sns
import pickle
import pandas as pd

# * Helper functions
def _to_df(x, y, label):
    return pd.DataFrame(dict(Fx=np.asarray(x), Fy=np.asarray(y), contact=label))

def trim(data, q=5):
    data = np.array(data)
    lo, hi = np.percentile(data, [q, 100-q])
    return data[(data > lo) & (data < hi)]           


class HumanoidVanillaRecorder(VideoRecorder):
    """Recorder for HumanoidFullModular environment."""
    # ----------------------------------------------------------------
    # Each entry:  <key str>  ->  function(tensor) -> log buffer
    # Child classes add to / override this dict in their __init__.
    # ----------------------------------------------------------------
    _HANDLERS: Dict[str, Callable[[torch.Tensor,'HumanoidVanillaRecorder'], None]] = {}

    # =================================================================
    # API to register new handlers
    # =================================================================
    @classmethod
    def register(cls, key: str):
        """Decorator:  @BaseRecorder.register("my_key") above a function."""
        def _decorator(fn):
            cls._HANDLERS[key] = fn
            return fn
        return _decorator
                                  
    def __init__(self, fps: int = 50, total_weight: int = 0, robot_cfg: ArticulationCfg = None):
        super().__init__(fps)
        self.log_buffer: dict[str, list] = defaultdict(list)
        self.episode_length = 0
        self.total_weight = total_weight
        self.robot_cfg = robot_cfg
        self.leg_joint_names = ['right_hip_yaw', 'right_hip_abad', 'right_hip_pitch', 'right_knee', 'right_ankle',
                                'left_hip_yaw', 'left_hip_abad', 'left_hip_pitch', 'left_knee', 'left_ankle']
        self.arm_joint_names = ['right_shoulder_pitch', 'right_shoulder_abad', 'right_shoulder_yaw', 'right_elbow',
                                'left_shoulder_pitch', 'left_shoulder_abad', 'left_shoulder_yaw', 'left_elbow']

    def __str__(self) -> str:
        """Return the string representation of the recorder."""
        msg = "HumanoidVanillaRecorder"
        msg += f" (fps: {self.fps})"
        return msg

    def log(self, image: np.ndarray | None, log_dict: dict[str, torch.Tensor] = dict()):
        """Log images and data."""
        super().log(image)

        # * Log data
        for key, tensor in log_dict.items():
            handler = self._HANDLERS.get(key)
            handler(tensor, self)

        self.episode_length += 1

    def save(self, resume_path: str):
        """Save the video and data."""
        super().save(resume_path)
        self.save_plot_velocity_tracking()
        self.save_plot_CM()
        self.save_plot_CM_bf()
        self.save_plot_dCM()
        self.save_plot_dCM_bf()
        self.save_plot_GRF()
        self.save_plot_GRF_distribution()
        self.save_plot_GRM()
        self.save_plot_GRM_distribution()
        self.save_plot_leg_joint_pos()
        self.save_plot_leg_joint_vel()
        self.save_plot_leg_joint_torque()
        self.save_plot_gait_sequence()
        self.save_plot_foot_velocity()
        self.save_log_buffer()
        self.save_RS_full_state()

    def save_plot_velocity_tracking(self):
        """ Make plot for velocity tracking """
        print("Creating plot for velocity tracking...")
        fig, axes = plt.subplots(3, 1, figsize=(9, 12))

        colors = ['r', 'g', 'b']

        axes[0].plot(np.arange(self.episode_length), self.log_buffer['root_lin_vel_b_x'], color=colors[0], label='Base Linear Velocity x')
        axes[0].plot(np.arange(self.episode_length), self.log_buffer['vel_command_x'], color=colors[0], linestyle='--', label='Velocity Command x')
        axes[0].set_title('Root Linear Velocity x')
        axes[0].set_ylabel('Velocity (m/s)')
        axes[0].legend()
        axes[1].plot(np.arange(self.episode_length), self.log_buffer['root_lin_vel_b_y'], color=colors[1], label='Base Linear Velocity y')
        axes[1].plot(np.arange(self.episode_length), self.log_buffer['vel_command_y'], color=colors[1], linestyle='--', label='Velocity Command y')
        axes[1].set_title('Root Linear Velocity y')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].legend()
        axes[2].plot(np.arange(self.episode_length), self.log_buffer['root_ang_vel_b_z'], color=colors[2], label='Base Angular Velocity z')
        axes[2].plot(np.arange(self.episode_length), self.log_buffer['vel_command_z'], color=colors[2], linestyle='--', label='Velocity Command z')
        axes[2].set_title('Root Angular Velocity z')
        axes[2].set_ylabel('Velocity (rad/s)')
        axes[2].set_xlabel('time (s)')
        axes[2].legend()

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_velocity_tracking_plot.pdf")
        plt.savefig(filepath)
        print(f"Velocity tracking plot saved to {filepath}")

    def save_plot_CM(self):
        """ Make plot for Centroidal Momentum """
        print("Creating plot for Centroidal Momentum...")
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))

        colors = ['k', 'g', 'b', 'r', 'gray']

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_base_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_leg_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_arm_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_des_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,0].legend()
        axes[0,0].set_title('CLM x')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_y'], color=colors[0], linewidth=2)
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_base_y'], color=colors[1], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_leg_y'], color=colors[2], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_arm_y'], color=colors[3], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_des_y'], color=colors[4], linestyle='--')
        axes[1,0].set_title('CLM y')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_z'], color=colors[0], linewidth=2)
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_base_z'], color=colors[1], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_leg_z'], color=colors[2], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_arm_z'], color=colors[3], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_des_z'], color=colors[4], linestyle='--')
        axes[2,0].set_title('CLM z')
        axes[2,0].set_xlabel('time (s)')

        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_base_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_leg_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_arm_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_des_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,1].legend()
        axes[0,1].set_title('CAM x')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_y'], color=colors[0], linewidth=2)
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_base_y'], color=colors[1], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_leg_y'], color=colors[2], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_arm_y'], color=colors[3], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_des_y'], color=colors[4], linestyle='--')
        axes[1,1].set_title('CAM y')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_z'], color=colors[0], linewidth=2)
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_base_z'], color=colors[1], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_leg_z'], color=colors[2], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_arm_z'], color=colors[3], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_des_z'], color=colors[4], linestyle='--')
        axes[2,1].set_title('CAM z')
        axes[2,1].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_CM_plot.pdf")
        plt.savefig(filepath)
        print(f"Centroidal Momentum plot saved to {filepath}")

    def save_plot_CM_bf(self):
        """ Make plot for Centroidal Momentum in base frame """
        print("Creating plot for Centroidal Momentum in base frame...")
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))

        colors = ['k', 'g', 'b', 'r', 'gray']

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_bf_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_base_bf_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_leg_bf_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_arm_bf_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_des_bf_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,0].legend()
        axes[0,0].set_title('CLM bf x')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_bf_y'], color=colors[0], linewidth=2)
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_base_bf_y'], color=colors[1], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_leg_bf_y'], color=colors[2], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_arm_bf_y'], color=colors[3], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_des_bf_y'], color=colors[4], linestyle='--')
        axes[1,0].set_title('CLM bf y')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_bf_z'], color=colors[0], linewidth=2)
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_base_bf_z'], color=colors[1], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_leg_bf_z'], color=colors[2], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_arm_bf_z'], color=colors[3], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['CLM_des_bf_z'], color=colors[4], linestyle='--')
        axes[2,0].set_title('CLM bf z')
        axes[2,0].set_xlabel('time (s)')

        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_bf_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_base_bf_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_leg_bf_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_arm_bf_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_des_bf_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,1].legend()
        axes[0,1].set_title('CAM bf x')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_bf_y'], color=colors[0], linewidth=2)
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_base_bf_y'], color=colors[1], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_leg_bf_y'], color=colors[2], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_arm_bf_y'], color=colors[3], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_des_bf_y'], color=colors[4], linestyle='--')
        axes[1,1].set_title('CAM bf y')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_bf_z'], color=colors[0], linewidth=2)
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_base_bf_z'], color=colors[1], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_leg_bf_z'], color=colors[2], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_arm_bf_z'], color=colors[3], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['CAM_des_bf_z'], color=colors[4], linestyle='--')
        axes[2,1].set_title('CAM bf z')
        axes[2,1].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_CM_bf_plot.pdf")
        plt.savefig(filepath)
        print(f"Centroidal Momentum in base frame plot saved to {filepath}")

    def save_plot_dCM(self):
        """ Make plot for the time derivative of Centroidal Momentum """
        print("Creating plot for the time derivative of Centroidal Momentum...")
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))

        colors = ['k', 'g', 'b', 'r', 'gray']

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_base_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_leg_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_arm_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_des_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,0].legend()
        axes[0,0].set_title('dCLM x')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_y'], color=colors[0], linewidth=2)
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_base_y'], color=colors[1], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_leg_y'], color=colors[2], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_arm_y'], color=colors[3], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_des_y'], color=colors[4], linestyle='--')
        axes[1,0].set_title('dCLM y')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_z'], color=colors[0], linewidth=2)
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_base_z'], color=colors[1], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_leg_z'], color=colors[2], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_arm_z'], color=colors[3], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_des_z'], color=colors[4], linestyle='--')
        axes[2,0].set_title('dCLM z')
        axes[2,0].set_xlabel('time (s)')

        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_base_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_leg_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_arm_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_des_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,1].legend()
        axes[0,1].set_title('dCAM x')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_y'], color=colors[0], linewidth=2)
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_base_y'], color=colors[1], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_leg_y'], color=colors[2], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_arm_y'], color=colors[3], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_des_y'], color=colors[4], linestyle='--')
        axes[1,1].set_title('dCAM y')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_z'], color=colors[0], linewidth=2)
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_base_z'], color=colors[1], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_leg_z'], color=colors[2], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_arm_z'], color=colors[3], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_des_z'], color=colors[4], linestyle='--')
        axes[2,1].set_title('dCAM z')
        axes[2,1].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_dCM_plot.pdf")
        plt.savefig(filepath)
        print(f"Time Derivative of Centroidal Momentum plot saved to {filepath}")

    def save_plot_dCM_bf(self):
        """ Make plot for the time derivative of Centroidal Momentum in base frame """
        print("Creating plot for the time derivative of Centroidal Momentum in base frame...")
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))

        colors = ['k', 'g', 'b', 'r', 'gray']

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_bf_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_base_bf_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_leg_bf_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_arm_bf_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_des_bf_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,0].legend()
        axes[0,0].set_title('dCLM bf x')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_bf_y'], color=colors[0], linewidth=2)
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_base_bf_y'], color=colors[1], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_leg_bf_y'], color=colors[2], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_arm_bf_y'], color=colors[3], linestyle='--')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_des_bf_y'], color=colors[4], linestyle='--')
        axes[1,0].set_title('dCLM bf y')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_bf_z'], color=colors[0], linewidth=2)
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_base_bf_z'], color=colors[1], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_leg_bf_z'], color=colors[2], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_arm_bf_z'], color=colors[3], linestyle='--')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['dCLM_des_bf_z'], color=colors[4], linestyle='--')
        axes[2,0].set_title('dCLM bf z')
        axes[2,0].set_xlabel('time (s)')

        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_bf_x'], color=colors[0], linewidth=2, label='Total')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_base_bf_x'], color=colors[1], linestyle='--', label='Base')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_leg_bf_x'], color=colors[2], linestyle='--', label='Leg')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_arm_bf_x'], color=colors[3], linestyle='--', label='Arm')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_des_bf_x'], color=colors[4], linestyle='--', label='Desired')
        axes[0,1].legend()
        axes[0,1].set_title('dCAM bf x')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_bf_y'], color=colors[0], linewidth=2)
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_base_bf_y'], color=colors[1], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_leg_bf_y'], color=colors[2], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_arm_bf_y'], color=colors[3], linestyle='--')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_des_bf_y'], color=colors[4], linestyle='--')
        axes[1,1].set_title('dCAM bf y')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_bf_z'], color=colors[0], linewidth=2)
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_base_bf_z'], color=colors[1], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_leg_bf_z'], color=colors[2], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_arm_bf_z'], color=colors[3], linestyle='--')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['dCAM_des_bf_z'], color=colors[4], linestyle='--')
        axes[2,1].set_title('dCAM bf z')
        axes[2,1].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_dCM_bf_plot.pdf")
        plt.savefig(filepath)
        print(f"Time Derivative of Centroidal Momentum in base frame plot saved to {filepath}")

    def save_plot_GRF(self):
        """ Make plot for Ground Reaction Force for right and left toe / heel"""
        print("Creating plot for Ground Reaction Force...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        colors = ['r', 'g', 'b', 'k']

        top_ylim = self.total_weight + 50

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['lf_toe_GRF_x'], color=colors[0], linewidth=2, label=r'F_x')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['lf_toe_GRF_y'], color=colors[1], linewidth=2, label=r'F_y')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['lf_toe_GRF_z'], color=colors[2], linewidth=2, label=r'F_z')
        axes[0,0].axhline(self.total_weight, color=colors[3], linestyle='--', label='Total Weight')
        axes[0,0].legend()
        axes[0,0].set_title('Left Toe GRF')
        axes[0,0].set_ylabel('Force (N)')
        axes[0,0].set_ylim(top=top_ylim)
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['lf_heel_GRF_x'], color=colors[0], linewidth=2, label=r'F_x')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['lf_heel_GRF_y'], color=colors[1], linewidth=2, label=r'F_y')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['lf_heel_GRF_z'], color=colors[2], linewidth=2, label=r'F_z')
        axes[1,0].axhline(self.total_weight, color=colors[3], linestyle='--', label='Total Weight')
        axes[1,0].legend()
        axes[1,0].set_title('Left Heel GRF')
        axes[1,0].set_xlabel('time (s)')
        axes[1,0].set_ylabel('Force (N)')
        axes[1,0].set_ylim(top=top_ylim)
        
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['rf_toe_GRF_x'], color=colors[0], linewidth=2, label=r'F_x')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['rf_toe_GRF_y'], color=colors[1], linewidth=2, label=r'F_y')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['rf_toe_GRF_z'], color=colors[2], linewidth=2, label=r'F_z')
        axes[0,1].axhline(self.total_weight, color=colors[3], linestyle='--', label='Total Weight')
        axes[0,1].legend()
        axes[0,1].set_title('Right Toe GRF')
        axes[0,1].set_ylabel('Force (N)')
        axes[0,1].set_ylim(top=top_ylim)
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['rf_heel_GRF_x'], color=colors[0], linewidth=2, label=r'F_x')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['rf_heel_GRF_y'], color=colors[1], linewidth=2, label=r'F_y')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['rf_heel_GRF_z'], color=colors[2], linewidth=2, label=r'F_z')
        axes[1,1].axhline(self.total_weight, color=colors[3], linestyle='--', label='Total Weight')
        axes[1,1].legend()
        axes[1,1].set_title('Right Heel GRF')
        axes[1,1].set_xlabel('time (s)')
        axes[1,1].set_ylabel('Force (N)')
        axes[1,1].set_ylim(top=top_ylim)

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_GRF_plot.pdf")
        plt.savefig(filepath)
        print(f"Ground Reaction Force plot saved to {filepath}")

    def save_plot_GRF_distribution(self):
        """ Make plot for Ground Reaction Force distribution for right and left toe / heel"""
    
        print("Creating plot for Ground Reaction Force distribution histogram / KDE estimation...")
        fig, axes = plt.subplots(3, 1, figsize=(9, 12))

        labels = ["RF Toe", "RF Heel", "LF Toe", "LF Heel"]
        colors = ["#d33c32", "#f4a582", "#0571b0", "#92c5de"]
        bins = 50

        sns.histplot(trim(self.log_buffer["rf_toe_GRF_x"]), bins=bins, kde=True, stat="density", ax=axes[0], color=colors[0], label=labels[0])
        sns.histplot(trim(self.log_buffer["rf_heel_GRF_x"]), bins=bins, kde=True, stat="density", ax=axes[0], color=colors[1], label=labels[1])
        sns.histplot(trim(self.log_buffer["lf_toe_GRF_x"]), bins=bins, kde=True, stat="density", ax=axes[0], color=colors[2], label=labels[2])
        sns.histplot(trim(self.log_buffer["lf_heel_GRF_x"]), bins=bins, kde=True, stat="density", ax=axes[0], color=colors[3], label=labels[3])
        axes[0].set_title("Ground Reaction Force $F_x$ [N]")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        sns.histplot(trim(self.log_buffer["rf_toe_GRF_y"]), bins=bins, kde=True, stat="density", ax=axes[1], color=colors[0], label=labels[0])
        sns.histplot(trim(self.log_buffer["rf_heel_GRF_y"]), bins=bins, kde=True, stat="density", ax=axes[1], color=colors[1], label=labels[1])
        sns.histplot(trim(self.log_buffer["lf_toe_GRF_y"]), bins=bins, kde=True, stat="density", ax=axes[1], color=colors[2], label=labels[2])
        sns.histplot(trim(self.log_buffer["lf_heel_GRF_y"]), bins=bins, kde=True, stat="density", ax=axes[1], color=colors[3], label=labels[3])
        axes[1].set_title("Ground Reaction Force $F_y$ [N]")
        axes[1].set_ylabel("Count")

        sns.histplot(trim(self.log_buffer["rf_toe_GRF_z"]), bins=bins, kde=True, stat="density", ax=axes[2], color=colors[0], label=labels[0])
        sns.histplot(trim(self.log_buffer["rf_heel_GRF_z"]), bins=bins, kde=True, stat="density", ax=axes[2], color=colors[1], label=labels[1])
        sns.histplot(trim(self.log_buffer["lf_toe_GRF_z"]), bins=bins, kde=True, stat="density", ax=axes[2], color=colors[2], label=labels[2])
        sns.histplot(trim(self.log_buffer["lf_heel_GRF_z"]), bins=bins, kde=True, stat="density", ax=axes[2], color=colors[3], label=labels[3])
        axes[2].set_title("Ground Reaction Force $F_z$ [N]")
        axes[2].set_ylabel("Count")

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_GRF_distribution_plot.pdf")
        plt.savefig(filepath)
        print(f"Ground Reaction Force distribution Histogram plot saved to {filepath}")

        print("Creating KDE plot for Ground-Reaction-Force distributions ...")
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)

        # ---------- F_x --------------------------------------------------------
        for dat, c, lab in zip(
            [self.log_buffer["rf_toe_GRF_x"],
            self.log_buffer["rf_heel_GRF_x"],
            self.log_buffer["lf_toe_GRF_x"],
            self.log_buffer["lf_heel_GRF_x"]],
            colors, labels):
            sns.kdeplot(trim(dat), ax=axes[0], color=c, linewidth=2, label=lab)

        axes[0].set_title(r"Ground-Reaction Force $F_x$ [N]")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # ---------- F_y --------------------------------------------------------
        for dat, c in zip(
            [self.log_buffer["rf_toe_GRF_y"],
            self.log_buffer["rf_heel_GRF_y"],
            self.log_buffer["lf_toe_GRF_y"],
            self.log_buffer["lf_heel_GRF_y"]], colors):
            sns.kdeplot(trim(dat), ax=axes[1], color=c, linewidth=2)

        axes[1].set_title(r"Ground-Reaction Force $F_y$ [N]")
        axes[1].set_ylabel("Density")

        # ---------- F_z --------------------------------------------------------
        for dat, c in zip(
            [self.log_buffer["rf_toe_GRF_z"],
            self.log_buffer["rf_heel_GRF_z"],
            self.log_buffer["lf_toe_GRF_z"],
            self.log_buffer["lf_heel_GRF_z"]], colors):
            sns.kdeplot(trim(dat), ax=axes[2], color=c, linewidth=2)

        axes[2].set_title(r"Ground-Reaction Force $F_z$ [N]")
        axes[2].set_ylabel("Density")
        axes[2].set_xlabel("Force [N]")

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_GRF_distribution_KDE_plot.pdf")
        plt.savefig(filepath)
        print(f"Ground Reaction Force distribution KDE plot saved to {filepath}")

        print("Creating 2D contour plot for Ground-Reaction-Force distributions ...")
        fig, ax = plt.subplots(figsize=(8, 6))

        # ------------------- build tidy table ---------------------
        df = pd.concat([
            _to_df(self.log_buffer["rf_toe_GRF_x"],  self.log_buffer["rf_toe_GRF_y"],  labels[0]),
            _to_df(self.log_buffer["rf_heel_GRF_x"], self.log_buffer["rf_heel_GRF_y"], labels[1]),
            _to_df(self.log_buffer["lf_toe_GRF_x"],  self.log_buffer["lf_toe_GRF_y"],  labels[2]),
            _to_df(self.log_buffer["lf_heel_GRF_x"], self.log_buffer["lf_heel_GRF_y"], labels[3]),
        ], ignore_index=True)

        # # optional: drop outer 2 % tails to avoid skewing axes
        qlo, qhi = df[['Fx', 'Fy']].quantile([0.02, 0.98]).values
        df = df[(df.Fx.between(qlo[0], qhi[0])) & (df.Fy.between(qlo[1], qhi[1]))]
        # ------------------- plot ---------------------------------
        palette = dict(zip(labels, colors))

        sns.kdeplot(data=df, x="Fx", y="Fy",
                    hue="contact", fill=True,
                    thresh=0.05,  # hide very outer contour
                    alpha=0.6,    # transparency of fill
                    palette=palette,
                    ax=ax)

        ax.set_xlabel(r"$F_x$  [N]")
        ax.set_ylabel(r"$F_y$  [N]")
        ax.set_title("2-D KDE of horizontal GRF")
        fig.tight_layout()

        save_path = os.path.join(self.folderpath, f"{self.checkpoint}_FxFy_kde.pdf")
        fig.savefig(save_path)
        print("Saved 2-D contour plot to", save_path)

        print("Creating combined contour plot for horizontal GRF distributions ...")
        # ------------------------------------------------------------------
        # 1) gather every Fx-Fy sample into a single DataFrame
        # ------------------------------------------------------------------
        fx = np.concatenate([
                self.log_buffer["rf_toe_GRF_x"],
                self.log_buffer["rf_heel_GRF_x"],
                self.log_buffer["lf_toe_GRF_x"],
                self.log_buffer["lf_heel_GRF_x"]
            ])
        fy = np.concatenate([
                self.log_buffer["rf_toe_GRF_y"],
                self.log_buffer["rf_heel_GRF_y"],
                self.log_buffer["lf_toe_GRF_y"],
                self.log_buffer["lf_heel_GRF_y"]
            ])
        df_xy = pd.DataFrame(dict(Fx=fx, Fy=fy))

        # optional: 5 % tail trimming for prettier axes
        lo, hi = df_xy[["Fx","Fy"]].quantile([0.02, 0.98]).values
        df_xy = df_xy[(df_xy.Fx.between(lo[0], hi[0])) &
                      (df_xy.Fy.between(lo[1], hi[1]))]

        # ------------------------------------------------------------------
        # 2) single 2-D KDE (no hue)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5,4))
        sns.kdeplot(data=df_xy, x="Fx", y="Fy",
                    fill=True, cmap="Blues",   # choose any cmap
                    alpha=0.6, thresh=0.05,    # thresh trims faint outer level
                    ax=ax)

        ax.set_xlabel(r"$F_x$  [N]")
        ax.set_ylabel(r"$F_y$  [N]")
        ax.set_title("Combined 2-D KDE of horizontal GRF")
        fig.tight_layout()

        save_path = os.path.join(self.folderpath,
                                f"{self.checkpoint}_FxFy_combined.pdf")
        fig.savefig(save_path)
        print("Saved combined contour plot to", save_path)

    def save_plot_GRM(self):
        """ Make plot for Ground Reaction Moment for right and left toe / heel"""
        print("Creating plot for Ground Reaction Moment...")
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))

        colors = ['r', 'g', 'b', 'k']

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['lf_GRM_x'], color=colors[0], linewidth=2, label=r'M_x')
        axes[0,0].set_title('Left GRM x')
        axes[0,0].set_ylabel('Moment (Nm)')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['lf_GRM_y'], color=colors[1], linewidth=2, label=r'M_y')
        axes[1,0].set_title('Left GRM y')
        axes[1,0].set_ylabel('Moment (Nm)')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['lf_GRM_z'], color=colors[2], linewidth=2, label=r'M_z')
        axes[2,0].set_title('Left GRM z')
        axes[2,0].set_ylabel('Moment (Nm)')
        axes[2,0].set_xlabel('time (s)')
        axes[2,0].set_ylim(bottom=-15, top=15)

        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['rf_GRM_x'], color=colors[0], linewidth=2, label=r'M_x')
        axes[0,1].set_title('Right GRM x')
        axes[0,1].set_ylabel('Moment (Nm)')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['rf_GRM_y'], color=colors[1], linewidth=2, label=r'M_y')
        axes[1,1].set_title('Right GRM y')
        axes[1,1].set_ylabel('Moment (Nm)')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['rf_GRM_z'], color=colors[2], linewidth=2, label=r'M_z')
        axes[2,1].set_title('Right GRM z')
        axes[2,1].set_ylabel('Moment (Nm)')
        axes[2,1].set_xlabel('time (s)')
        axes[2,1].set_ylim(bottom=-6, top=6)

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_GRM_plot.pdf")
        plt.savefig(filepath)
        print(f"Ground Reaction Moment plot saved to {filepath}")

    def save_plot_GRM_distribution(self):
        """ Make plot for Ground Reaction Moment distribution for right and left foot"""
        print("Creating plot for Ground Reaction Moment distribution histogram / KDE estimation...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        labels = ["Left", "Right"]
        colors = ["#0571b0", "#d33c32"]

        sns.kdeplot(trim(self.log_buffer["lf_GRM_z"]), ax=axes[0], color=colors[0], linewidth=2, label=labels[0])
        sns.kdeplot(trim(self.log_buffer["rf_GRM_z"]), ax=axes[1], color=colors[1], linewidth=2, label=labels[1])
        axes[0].set_title("Ground Reaction Moment $M_z$ [Nm]")
        axes[0].set_ylabel("Density")
        axes[0].set_xlabel("Moment [Nm]")
        axes[0].legend()

        axes[1].set_title(r"Ground-Reaction Moment $M_z$ [Nm]")
        axes[1].set_ylabel("Density")
        axes[1].set_xlabel("Moment [Nm]")
        axes[1].legend()

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_GRM_distribution_plot.pdf")
        plt.savefig(filepath)
        print(f"Ground Reaction Moment distribution KDE plot saved to {filepath}")

    def save_plot_leg_joint_pos(self):
        """ Make plot for leg joint position """
        print("Creating plot for leg joint position...")
        fig, axes = plt.subplots(5, 2, figsize=(12, 18))

        for j in range(2):
            for i in range(5):
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'leg_joint_pos_{i+5*j}'], linewidth=2, label='Actual')
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'leg_joint_pos_target_{i+5*j}'], linestyle='--', label='Target')
                axes[i, j].legend()
                axes[i, j].set_title(f'Joint {self.leg_joint_names[i+5*j]} Position')
                axes[i, j].set_ylabel('Position (rad)')
                axes[i, j].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_leg_joint_pos_plot.pdf")
        plt.savefig(filepath)
        print(f"Leg Joint Position plot saved to {filepath}")

    def save_plot_leg_joint_vel(self):
        """ Make plot for leg joint velocity """
        print("Creating plot for leg joint velocity...")
        fig, axes = plt.subplots(5, 2, figsize=(12, 18))

        for j in range(2):
            for i in range(5):
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'leg_joint_vel_{i+5*j}'], linewidth=2)
                axes[i, j].set_title(f'Joint {self.leg_joint_names[i+5*j]} Velocity')
                axes[i, j].set_ylabel('Velocity (rad/s)')
                axes[i, j].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_leg_joint_vel_plot.pdf")
        plt.savefig(filepath)
        print(f"Leg Joint Velocity plot saved to {filepath}")

    def save_plot_leg_joint_torque(self):
        """ Make plot for leg joint torque """
        print("Creating plot for leg joint torque...")
        fig, axes = plt.subplots(5, 2, figsize=(12, 18))

        for j in range(2):
            for i in range(5):
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'leg_joint_torque_{i+5*j}'], linewidth=2)
                axes[i, j].set_title(f'Joint {self.leg_joint_names[i+5*j]} Torque')
                axes[i, j].set_ylabel('Torque (Nm)')
                axes[i, j].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_leg_joint_torque_plot.pdf")
        plt.savefig(filepath)
        print(f"Leg Joint Torque plot saved to {filepath}")

    def save_plot_gait_sequence(self):
        """ Make plot for gait sequence """
        print("Creating plot for gait sequencee...")
        fig, ax = plt.subplots(figsize=(8, 4))

        foot_names = ['Right', 'Left']
        COLORS = ['tab:pink', 'tab:cyan']
        CMD_COLORS = ['red', 'blue']
        threshold = 0.5  # Threshold for contact
        phase = np.asarray(self.log_buffer["phase"])

        # Plot the ground truth gait sequence
        gt_gait_data = np.array([(phase < threshold), 
                                 (phase >= threshold)], dtype=int)
        
        for idx, (leg_data, color) in enumerate(zip(gt_gait_data, CMD_COLORS)):
            # Find the start points and durations of the stance phases
            stance_starts = np.where(np.diff(leg_data, prepend=0, append=0) == 1)[0]
            stance_ends = np.where(np.diff(leg_data, prepend=0, append=0) == -1)[0]

            durations = stance_ends - stance_starts
            
            # Create a series of broken horizontal bars for each leg
            bars = [(start, duration) for start, duration in zip(stance_starts, durations)]
            ax.broken_barh(bars, (idx - 0.4, 0.8), facecolors=color, alpha=0.6, label=f'Desired {foot_names[idx]} Step')

        # Plot the actual gait sequence
        gait_data = np.array([np.array(self.log_buffer["right_foot_contact"], dtype=int), 
                              np.array(self.log_buffer["left_foot_contact"], dtype=int)])

        for idx, (leg_data, color) in enumerate(zip(gait_data, COLORS)):
            # Find the start points and durations of the stance phases
            stance_starts = np.where(np.diff(leg_data, prepend=0, append=0) == 1)[0]
            stance_ends = np.where(np.diff(leg_data, prepend=0, append=0) == -1)[0]

            durations = stance_ends - stance_starts
            
            # Create a series of broken horizontal bars for each leg
            bars = [(start, duration) for start, duration in zip(stance_starts, durations)]
            ax.broken_barh(bars, (idx - 0.4, 0.8), facecolors=color, alpha=0.6, label=f'Measured {foot_names[idx]} Step')

        # Set the y-ticks to be in the middle of each leg's bar
        ax.set_yticks(np.arange(gt_gait_data.shape[0]))
        ax.set_yticklabels(['Right', 'Left'])

        # Set the limits and labels
        ax.set_ylim(-1, gt_gait_data.shape[0])
        ax.set_xlim(0, gt_gait_data.shape[1])
        ax.set_xlabel('Episode Length')
        ax.set_title('Gait Sequence')

        # Get handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Define the desired order of your handles and labels, by indices
        new_order = [1, 0, 3, 2]  # for example, to put the fourth item first, etc.

        # Reorder handles and labels
        ordered_handles = [handles[idx] for idx in new_order]
        ordered_labels = [labels[idx] for idx in new_order]

        # Create the new legend with the reordered handles and labels
        ax.legend(ordered_handles, ordered_labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0))

        # Remove the spines on the top and right sides
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_gait_sequence_plot.pdf")
        plt.savefig(filepath)
        print(f"Gait Sequence plot saved to {filepath}")

    def save_plot_foot_velocity(self):
        """ Make plot for foot velocity """
        print("Creating plot for foot velocity...")
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))

        colors = ['r', 'g', 'b']

        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['contact_schedule'], color='k', linewidth=2, label='Contact Schedule')
        axes[0,0].plot(np.arange(self.episode_length), self.log_buffer['left_foot_velocity_x'], color=colors[0], linewidth=2, label='Left Foot Vel X')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['contact_schedule'], color='k', linewidth=2, label='Contact Schedule')
        axes[1,0].plot(np.arange(self.episode_length), self.log_buffer['left_foot_velocity_y'], color=colors[1], linewidth=2, label='Left Foot Vel Y')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['contact_schedule'], color='k', linewidth=2, label='Contact Schedule')
        axes[2,0].plot(np.arange(self.episode_length), self.log_buffer['left_foot_velocity_z'], color=colors[2], linewidth=2, label='Left Foot Vel Z')
        axes[0,0].set_title('Left Foot Velocity')
        axes[0,0].set_ylabel('Velocity (m/s)')
        axes[1,0].set_ylabel('Velocity (m/s)')
        axes[2,0].set_ylabel('Velocity (m/s)')
        axes[2,0].set_xlabel('time (s)')
        
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['contact_schedule'], color='k', linewidth=2, label='Contact Schedule')
        axes[0,1].plot(np.arange(self.episode_length), self.log_buffer['right_foot_velocity_x'], color=colors[0], linewidth=2, label='Right Foot Vel X')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['contact_schedule'], color='k', linewidth=2, label='Contact Schedule')
        axes[1,1].plot(np.arange(self.episode_length), self.log_buffer['right_foot_velocity_y'], color=colors[1], linewidth=2, label='Right Foot Vel Y')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['contact_schedule'], color='k', linewidth=2, label='Contact Schedule')
        axes[2,1].plot(np.arange(self.episode_length), self.log_buffer['right_foot_velocity_z'], color=colors[2], linewidth=2, label='Right Foot Vel Z')
        axes[0,1].set_title('Right Foot Velocity')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[1,1].set_ylabel('Velocity (m/s)')
        axes[2,1].set_ylabel('Velocity (m/s)')
        axes[2,1].set_xlabel('time (s)')

        for ax in axes.flat:
            ax.legend()

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_foot_velocity_plot.pdf")
        plt.savefig(filepath)
        print(f"Foot Velocity plot saved to {filepath}")

    def save_log_buffer(self):
        """ Save the log buffer to a pickle file """
        print("Saving log buffer...")

        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_log_buffer.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.log_buffer), f)

        print(f"Log buffer saved to {filepath}")

    def save_RS_full_state(self):
        """ Save the full state of the robot """
        gen_coord = torch.stack(self.log_buffer["gen_coord"], dim=-1)
        gen_vel_body = torch.stack(self.log_buffer["gen_vel_body"], dim=-1)
        full_state = torch.cat((gen_coord, gen_vel_body), dim=1)

        print("Saving full state of the robot...")

        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_full_state.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(full_state.cpu().numpy(), f)

        print(f"Full state of the robot saved to {filepath}")

# ═════════════════════  register handlers  ═════════════════════ #

@HumanoidVanillaRecorder.register("root_lin_vel_b")
def _root_lin_vel_b_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["root_lin_vel_b_x"].append(t[0].item())
    rec.log_buffer["root_lin_vel_b_y"].append(t[1].item())
    rec.log_buffer["root_lin_vel_b_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("root_ang_vel_b")
def _root_ang_vel_b_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["root_ang_vel_b_x"].append(t[0].item())
    rec.log_buffer["root_ang_vel_b_y"].append(t[1].item())
    rec.log_buffer["root_ang_vel_b_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("vel_command")
def _vel_command_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["vel_command_x"].append(t[0].item())
    rec.log_buffer["vel_command_y"].append(t[1].item())
    rec.log_buffer["vel_command_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("CM")
def _CM_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_x"].append(t[0].item())
    rec.log_buffer["CLM_y"].append(t[1].item())
    rec.log_buffer["CLM_z"].append(t[2].item())
    rec.log_buffer["CAM_x"].append(t[3].item())
    rec.log_buffer["CAM_y"].append(t[4].item())
    rec.log_buffer["CAM_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_base")
def _CM_base_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_base_x"].append(t[0].item())
    rec.log_buffer["CLM_base_y"].append(t[1].item())
    rec.log_buffer["CLM_base_z"].append(t[2].item())
    rec.log_buffer["CAM_base_x"].append(t[3].item())
    rec.log_buffer["CAM_base_y"].append(t[4].item())
    rec.log_buffer["CAM_base_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_leg")
def _CM_leg_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_leg_x"].append(t[0].item())
    rec.log_buffer["CLM_leg_y"].append(t[1].item())
    rec.log_buffer["CLM_leg_z"].append(t[2].item())
    rec.log_buffer["CAM_leg_x"].append(t[3].item())
    rec.log_buffer["CAM_leg_y"].append(t[4].item())
    rec.log_buffer["CAM_leg_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_arm")
def _CM_arm_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_arm_x"].append(t[0].item())
    rec.log_buffer["CLM_arm_y"].append(t[1].item())
    rec.log_buffer["CLM_arm_z"].append(t[2].item())
    rec.log_buffer["CAM_arm_x"].append(t[3].item())
    rec.log_buffer["CAM_arm_y"].append(t[4].item())
    rec.log_buffer["CAM_arm_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_des")
def _CM_des_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_des_x"].append(t[0].item())
    rec.log_buffer["CLM_des_y"].append(t[1].item())
    rec.log_buffer["CLM_des_z"].append(t[2].item())
    rec.log_buffer["CAM_des_x"].append(t[3].item())
    rec.log_buffer["CAM_des_y"].append(t[4].item())
    rec.log_buffer["CAM_des_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_bf")
def _CM_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_bf_x"].append(t[0].item())
    rec.log_buffer["CLM_bf_y"].append(t[1].item())
    rec.log_buffer["CLM_bf_z"].append(t[2].item())
    rec.log_buffer["CAM_bf_x"].append(t[3].item())
    rec.log_buffer["CAM_bf_y"].append(t[4].item())
    rec.log_buffer["CAM_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_base_bf")
def _CM_base_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_base_bf_x"].append(t[0].item())
    rec.log_buffer["CLM_base_bf_y"].append(t[1].item())
    rec.log_buffer["CLM_base_bf_z"].append(t[2].item())
    rec.log_buffer["CAM_base_bf_x"].append(t[3].item())
    rec.log_buffer["CAM_base_bf_y"].append(t[4].item())
    rec.log_buffer["CAM_base_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_leg_bf")
def _CM_leg_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_leg_bf_x"].append(t[0].item())
    rec.log_buffer["CLM_leg_bf_y"].append(t[1].item())
    rec.log_buffer["CLM_leg_bf_z"].append(t[2].item())
    rec.log_buffer["CAM_leg_bf_x"].append(t[3].item())
    rec.log_buffer["CAM_leg_bf_y"].append(t[4].item())
    rec.log_buffer["CAM_leg_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_arm_bf")
def _CM_arm_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_arm_bf_x"].append(t[0].item())
    rec.log_buffer["CLM_arm_bf_y"].append(t[1].item())
    rec.log_buffer["CLM_arm_bf_z"].append(t[2].item())
    rec.log_buffer["CAM_arm_bf_x"].append(t[3].item())
    rec.log_buffer["CAM_arm_bf_y"].append(t[4].item())
    rec.log_buffer["CAM_arm_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("CM_des_bf")
def _CM_des_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["CLM_des_bf_x"].append(t[0].item())
    rec.log_buffer["CLM_des_bf_y"].append(t[1].item())
    rec.log_buffer["CLM_des_bf_z"].append(t[2].item())
    rec.log_buffer["CAM_des_bf_x"].append(t[3].item())
    rec.log_buffer["CAM_des_bf_y"].append(t[4].item())
    rec.log_buffer["CAM_des_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM")
def _dCM_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_x"].append(t[0].item())
    rec.log_buffer["dCLM_y"].append(t[1].item())
    rec.log_buffer["dCLM_z"].append(t[2].item())
    rec.log_buffer["dCAM_x"].append(t[3].item())
    rec.log_buffer["dCAM_y"].append(t[4].item())
    rec.log_buffer["dCAM_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_base")
def _dCM_base_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_base_x"].append(t[0].item())
    rec.log_buffer["dCLM_base_y"].append(t[1].item())
    rec.log_buffer["dCLM_base_z"].append(t[2].item())
    rec.log_buffer["dCAM_base_x"].append(t[3].item())
    rec.log_buffer["dCAM_base_y"].append(t[4].item())
    rec.log_buffer["dCAM_base_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_leg")
def _dCM_leg_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_leg_x"].append(t[0].item())
    rec.log_buffer["dCLM_leg_y"].append(t[1].item())
    rec.log_buffer["dCLM_leg_z"].append(t[2].item())
    rec.log_buffer["dCAM_leg_x"].append(t[3].item())
    rec.log_buffer["dCAM_leg_y"].append(t[4].item())
    rec.log_buffer["dCAM_leg_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_arm")
def _dCM_arm_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_arm_x"].append(t[0].item())
    rec.log_buffer["dCLM_arm_y"].append(t[1].item())
    rec.log_buffer["dCLM_arm_z"].append(t[2].item())
    rec.log_buffer["dCAM_arm_x"].append(t[3].item())
    rec.log_buffer["dCAM_arm_y"].append(t[4].item())
    rec.log_buffer["dCAM_arm_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_des")
def _dCM_des_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_des_x"].append(t[0].item())
    rec.log_buffer["dCLM_des_y"].append(t[1].item())
    rec.log_buffer["dCLM_des_z"].append(t[2].item())
    rec.log_buffer["dCAM_des_x"].append(t[3].item())
    rec.log_buffer["dCAM_des_y"].append(t[4].item())
    rec.log_buffer["dCAM_des_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_bf")
def _dCM_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_bf_x"].append(t[0].item())
    rec.log_buffer["dCLM_bf_y"].append(t[1].item())
    rec.log_buffer["dCLM_bf_z"].append(t[2].item())
    rec.log_buffer["dCAM_bf_x"].append(t[3].item())
    rec.log_buffer["dCAM_bf_y"].append(t[4].item())
    rec.log_buffer["dCAM_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_base_bf")
def _dCM_base_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_base_bf_x"].append(t[0].item())
    rec.log_buffer["dCLM_base_bf_y"].append(t[1].item())
    rec.log_buffer["dCLM_base_bf_z"].append(t[2].item())
    rec.log_buffer["dCAM_base_bf_x"].append(t[3].item())
    rec.log_buffer["dCAM_base_bf_y"].append(t[4].item())
    rec.log_buffer["dCAM_base_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_leg_bf")
def _dCM_leg_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_leg_bf_x"].append(t[0].item())
    rec.log_buffer["dCLM_leg_bf_y"].append(t[1].item())
    rec.log_buffer["dCLM_leg_bf_z"].append(t[2].item())
    rec.log_buffer["dCAM_leg_bf_x"].append(t[3].item())
    rec.log_buffer["dCAM_leg_bf_y"].append(t[4].item())
    rec.log_buffer["dCAM_leg_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_arm_bf")
def _dCM_arm_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_arm_bf_x"].append(t[0].item())
    rec.log_buffer["dCLM_arm_bf_y"].append(t[1].item())
    rec.log_buffer["dCLM_arm_bf_z"].append(t[2].item())
    rec.log_buffer["dCAM_arm_bf_x"].append(t[3].item())
    rec.log_buffer["dCAM_arm_bf_y"].append(t[4].item())
    rec.log_buffer["dCAM_arm_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("dCM_des_bf")
def _dCM_des_bf_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["dCLM_des_bf_x"].append(t[0].item())
    rec.log_buffer["dCLM_des_bf_y"].append(t[1].item())
    rec.log_buffer["dCLM_des_bf_z"].append(t[2].item())
    rec.log_buffer["dCAM_des_bf_x"].append(t[3].item())
    rec.log_buffer["dCAM_des_bf_y"].append(t[4].item())
    rec.log_buffer["dCAM_des_bf_z"].append(t[5].item())

@HumanoidVanillaRecorder.register("rf_toe_GRF")
def _rf_toe_GRF_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["rf_toe_GRF_x"].append(t[0].item())
    rec.log_buffer["rf_toe_GRF_y"].append(t[1].item())
    rec.log_buffer["rf_toe_GRF_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("rf_heel_GRF")
def _rf_heel_GRF_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["rf_heel_GRF_x"].append(t[0].item())
    rec.log_buffer["rf_heel_GRF_y"].append(t[1].item())
    rec.log_buffer["rf_heel_GRF_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("lf_toe_GRF")
def _lf_toe_GRF_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["lf_toe_GRF_x"].append(t[0].item())
    rec.log_buffer["lf_toe_GRF_y"].append(t[1].item())
    rec.log_buffer["lf_toe_GRF_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("lf_heel_GRF")
def _lf_heel_GRF_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["lf_heel_GRF_x"].append(t[0].item())
    rec.log_buffer["lf_heel_GRF_y"].append(t[1].item())
    rec.log_buffer["lf_heel_GRF_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("rf_GRM")
def _rf_GRM_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["rf_GRM_x"].append(t[0].item())
    rec.log_buffer["rf_GRM_y"].append(t[1].item())
    rec.log_buffer["rf_GRM_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("lf_GRM")
def _lf_GRM_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["lf_GRM_x"].append(t[0].item())
    rec.log_buffer["lf_GRM_y"].append(t[1].item())
    rec.log_buffer["lf_GRM_z"].append(t[2].item())

@HumanoidVanillaRecorder.register("leg_joint_pos")
def _leg_joint_pos_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"leg_joint_pos_{i}"].append(t[i].item())

@HumanoidVanillaRecorder.register("leg_joint_pos_target")
def _leg_joint_pos_target_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"leg_joint_pos_target_{i}"].append(t[i].item())

@HumanoidVanillaRecorder.register("leg_joint_vel")
def _leg_joint_vel_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"leg_joint_vel_{i}"].append(t[i].item())

@HumanoidVanillaRecorder.register("leg_joint_torque")
def _leg_joint_torque_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"leg_joint_torque_{i}"].append(t[i].item())

@HumanoidVanillaRecorder.register("phase")
def _phase_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    """Log the phase of the robot."""
    rec.log_buffer["phase"].append(t.item())

@HumanoidVanillaRecorder.register("foot_contact")
def _foot_contact_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    """Log the foot contact state of the robot."""
    rec.log_buffer["right_foot_contact"].append(t[0].item())
    rec.log_buffer["left_foot_contact"].append(t[1].item())

@HumanoidVanillaRecorder.register("contact_schedule")
def _contact_schedule_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    """Log the contact schedule of the robot."""
    rec.log_buffer["contact_schedule"].append(t.item())

@HumanoidVanillaRecorder.register("foot_velocity")
def _foot_velocity_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    """Log the foot_velocity of the robot."""
    rec.log_buffer["right_foot_velocity_x"].append(t[0, 0].item())
    rec.log_buffer["right_foot_velocity_y"].append(t[0, 1].item())
    rec.log_buffer["right_foot_velocity_z"].append(t[0, 2].item())
    rec.log_buffer["left_foot_velocity_x"].append(t[1, 0].item())
    rec.log_buffer["left_foot_velocity_y"].append(t[1, 1].item())
    rec.log_buffer["left_foot_velocity_z"].append(t[1, 2].item())

@HumanoidVanillaRecorder.register("gen_coord")
def _gen_coord_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["gen_coord"].append(t)

@HumanoidVanillaRecorder.register("gen_vel_body")
def _gen_vel_body_handler(t: torch.Tensor, rec: HumanoidVanillaRecorder):
    rec.log_buffer["gen_vel_body"].append(t)


class HumanoidFullVanillaRecorder(HumanoidVanillaRecorder):
    """Recorder for HumanoidFullVanilla environment."""

    def __init__(self, fps: int = 50, total_weight: int = 0, robot_cfg: ArticulationCfg = None):
        super().__init__(fps, total_weight, robot_cfg)

    def save(self, resume_path: str):
        """Save the video and data."""
        super().save(resume_path)
        self.save_plot_arm_joint_pos()
        self.save_plot_arm_joint_vel()
        self.save_plot_arm_joint_torque()

    def save_plot_arm_joint_pos(self):
        """ Make plot for arm joint position """
        print("Creating plot for arm joint position...")
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))

        for j in range(2):
            for i in range(4):
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'arm_joint_pos_{i+4*j}'], linewidth=2, label='Actual')
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'arm_joint_pos_target_{i+4*j}'], linestyle='--', label='Target')
                axes[i, j].legend()
                axes[i, j].set_title(f'Joint {self.arm_joint_names[i+4*j]} Position')
                axes[i, j].set_ylabel('Position (rad)')
                axes[i, j].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_arm_joint_pos_plot.pdf")
        plt.savefig(filepath)
        print(f"Arm Joint Position plot saved to {filepath}")

    def save_plot_arm_joint_vel(self):
        """ Make plot for arm joint velocity """
        print("Creating plot for arm joint velocity...")
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))

        for j in range(2):
            for i in range(4):
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'arm_joint_vel_{i+4*j}'], linewidth=2)
                axes[i, j].set_title(f'Joint {self.arm_joint_names[i+4*j]} Velocity')
                axes[i, j].set_ylabel('Velocity (rad/s)')
                axes[i, j].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_arm_joint_vel_plot.pdf")
        plt.savefig(filepath)
        print(f"Arm Joint Velocity plot saved to {filepath}")

    def save_plot_arm_joint_torque(self):
        """ Make plot for arm joint torque """
        print("Creating plot for arm joint torque...")
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))

        for j in range(2):
            for i in range(4):
                axes[i, j].plot(np.arange(self.episode_length), self.log_buffer[f'arm_joint_torque_{i+4*j}'], linewidth=2)
                axes[i, j].set_title(f'Joint {self.arm_joint_names[i+4*j]} Torque')
                axes[i, j].set_ylabel('Torque (Nm)')
                axes[i, j].set_xlabel('time (s)')

        plt.tight_layout()

        # * Save the plot
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}_arm_joint_torque_plot.pdf")
        plt.savefig(filepath)
        print(f"Arm Joint Torque plot saved to {filepath}")

# ═════════════════════  register handlers  ═════════════════════ #

@HumanoidFullVanillaRecorder.register("arm_joint_pos")
def _arm_joint_pos_handler(t: torch.Tensor, rec: HumanoidFullVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"arm_joint_pos_{i}"].append(t[i].item())

@HumanoidFullVanillaRecorder.register("arm_joint_pos_target")
def _arm_joint_pos_target_handler(t: torch.Tensor, rec: HumanoidFullVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"arm_joint_pos_target_{i}"].append(t[i].item())

@HumanoidFullVanillaRecorder.register("arm_joint_vel")
def _arm_joint_vel_handler(t: torch.Tensor, rec: HumanoidFullVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"arm_joint_vel_{i}"].append(t[i].item())

@HumanoidFullVanillaRecorder.register("arm_joint_torque")
def _arm_joint_torque_handler(t: torch.Tensor, rec: HumanoidFullVanillaRecorder):
    for i in range(t.shape[0]):
        rec.log_buffer[f"arm_joint_torque_{i}"].append(t[i].item())