# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator that does nothing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, \
                                    PINK_ARROW_X_MARKER_CFG, CYAN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from .commands_cfg import StepCommandCfg


class StepCommand(CommandTerm):
    """Command generator for generating step commands.
    """

    cfg: StepCommandCfg
    """Configuration for the command generator."""

    def __str__(self) -> str:
        msg = "StepCommand:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self):
        """ Return full step period."""
        return self._env.full_step_period

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float):
        pass

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            scale = (.07, .07, .15)
            marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/right_step_command"
            marker_cfg.markers["arrow"].scale = scale
            self.right_step_command_visualizer = VisualizationMarkers(marker_cfg)
            self.right_step_command_visualizer.set_visibility(True)

            marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/left_step_command"
            marker_cfg.markers["arrow"].scale = scale
            self.left_step_command_visualizer = VisualizationMarkers(marker_cfg)
            self.left_step_command_visualizer.set_visibility(True)

            marker_cfg = PINK_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/right_current_step"
            marker_cfg.markers["arrow"].scale = scale
            self.right_current_step_visualizer = VisualizationMarkers(marker_cfg)
            self.right_current_step_visualizer.set_visibility(True)

            marker_cfg = CYAN_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/left_current_step"
            marker_cfg.markers["arrow"].scale = scale
            self.left_current_step_visualizer = VisualizationMarkers(marker_cfg)
            self.left_current_step_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        """ Visualize step commands and current step location 
        """
        right_step_command_loc = torch.hstack((self._env.step_commands[:,0,0:1], self._env.step_commands[:,0,1:2], torch.zeros(self.num_envs,1, device=self.device)))
        left_step_command_loc = torch.hstack((self._env.step_commands[:,1,0:1], self._env.step_commands[:,1,1:2], torch.zeros(self.num_envs,1, device=self.device)))
        
        right_step_command_quat = math_utils.quat_from_euler_xyz(torch.zeros(self.num_envs, device=self.device),
                                                                 torch.zeros(self.num_envs, device=self.device),
                                                                 self._env.step_commands[:,0,2])
        left_step_command_quat = math_utils.quat_from_euler_xyz(torch.zeros(self.num_envs, device=self.device),
                                                                torch.zeros(self.num_envs, device=self.device),
                                                                self._env.step_commands[:,1,2])

        self.right_step_command_visualizer.visualize(right_step_command_loc, right_step_command_quat)
        self.left_step_command_visualizer.visualize(left_step_command_loc, left_step_command_quat)

        right_current_step_loc = torch.hstack((self._env.current_step[:,0,0:1], self._env.current_step[:,0,1:2], torch.zeros(self.num_envs,1, device=self.device)))
        left_current_step_loc = torch.hstack((self._env.current_step[:,1,0:1], self._env.current_step[:,1,1:2], torch.zeros(self.num_envs,1, device=self.device)))

        right_current_step_quat = math_utils.quat_from_euler_xyz(torch.zeros(self.num_envs, device=self.device),
                                                                 torch.zeros(self.num_envs, device=self.device),
                                                                 self._env.current_step[:,0,2])
        left_current_step_quat = math_utils.quat_from_euler_xyz(torch.zeros(self.num_envs, device=self.device),
                                                                torch.zeros(self.num_envs, device=self.device),
                                                                self._env.current_step[:,1,2])
        
        self.right_current_step_visualizer.visualize(right_current_step_loc, right_current_step_quat)
        self.left_current_step_visualizer.visualize(left_current_step_loc, left_current_step_quat)
