# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import VelocityCommandCfg


class VelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in the body / world frame from uniform distribution.

    The command comprises of a velocity in x and y direction and yaw velocity in the body / world frame.

    """

    cfg: VelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel in the body / world frame
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "VelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base / world frame. Shape is (num_envs, 3)."""
        return self.vel_command

    @command.setter
    def command(self, value):
        self.vel_command = value  # Allow setting the command attribute

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

        if self.cfg.reference == "body":
            # -- angular velocity - z direction
            self.vel_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # set small commands to zero
        self.vel_command[env_ids, :2] *= (torch.norm(self.vel_command[env_ids, :2], dim=1) > self.cfg.lin_vel_threshold).unsqueeze(1)
        self.vel_command[env_ids, 2] *= (torch.abs(self.vel_command[env_ids, 2]) > self.cfg.ang_vel_threshold)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        if self.cfg.reference == "body":
            self.metrics["error_vel_xy"] += (
                torch.norm(self.vel_command[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
            )
        else:
            self.metrics["error_vel_xy"] += (
                torch.norm(self.vel_command[:, :2] - self.robot.data.root_lin_vel_w[:, :2], dim=-1) / max_command_step
            )

    def _update_command(self):
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                self.base_yaw_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
                self.base_yaw_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_yaw_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
            self.base_yaw_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_yaw_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)
                self.base_yaw_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        if self.cfg.reference == "body":
            offset = torch.zeros_like(base_pos_w)
            offset[:, 2] = 0.5
            offset = math_utils.quat_apply(self.robot.data.root_quat_w, offset)
            base_pos_w += offset
        else:
            base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        yaw_vel_des_arrow_scale, yaw_vel_des_arrow_quat = self._resolve_yaw_velocity_to_arrow(self.command[:, 2])
        if self.cfg.reference == "body":
            vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        else:
            vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_w[:, :2])
        yaw_vel_arrow_scale, yaw_vel_arrow_quat = self._resolve_yaw_velocity_to_arrow(self.robot.data.root_ang_vel_b[:, 2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_yaw_vel_goal_visualizer.visualize(base_pos_w, yaw_vel_des_arrow_quat, yaw_vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.base_yaw_vel_visualizer.visualize(base_pos_w, yaw_vel_arrow_quat, yaw_vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY world velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        if self.cfg.reference == "body":
            # convert everything back from base to world frame
            base_quat_w = self.robot.data.root_quat_w
            arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    
    def _resolve_yaw_velocity_to_arrow(self, yaw_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the yaw velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(yaw_velocity.shape[0], 1)
        arrow_scale[:, 0] *= yaw_velocity * 2.0
        # arrow-direction
        zeros = torch.zeros_like(yaw_velocity)
        rotateY = -torch.pi/2 * torch.ones_like(yaw_velocity)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, rotateY, zeros)
        if self.cfg.reference == "body":
            # convert everything back from base to world frame
            base_quat_w = self.robot.data.root_quat_w
            arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat