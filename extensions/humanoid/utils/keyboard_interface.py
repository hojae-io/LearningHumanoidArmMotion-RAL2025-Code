# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) control."""

from __future__ import annotations

import numpy as np
import weakref
from collections.abc import Callable

import carb
import omni
import torch

import extensions.humanoid.mdp as brl_mdp
from extensions.utils.base_keyboard import BaseKeyboard

class VanillaKeyboard(BaseKeyboard):
    r"""A vanilla keyboard controller for basic functionality.

    This class is designed to provide a keyboard controller for mobile base (such as quadrupeds).
    It uses the Omniverse keyboard interface to listen to keyboard events.

    Key bindings:
        ====================== ================
        Basic Command          Key 
        ====================== ================
        Reset                  R       
        Screenshot             X
        Record                 W
        ====================== ================ ================
        Velocity Command       Key (+ve axis)   Key (-ve axis)
        ====================== ================ ================
        Move along x-axis      L                H
        Move along y-axis      K                J
        Rotate along z-axis    U                I
        ====================== ================ ================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, env, v_x_sensitivity: float = 0.5, v_y_sensitivity: float = 0.5, yaw_sensitivity: float = torch.pi/6):
        """Initialize the keyboard layer.

        Args:
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 0.2.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 0.2.
            yaw_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to pi/6.
        """
        super().__init__(env)
        # store inputs
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.yaw_sensitivity = yaw_sensitivity
        self.disturbance = -3.0

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"\tKeyboard Controller: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset : R\n"
        msg += "\tScreenshot : X\n"
        msg += "\tRecord : W\n"
        msg += "\tShoot : C\n"
        msg += "\tForce Disturbance : F\n"
        msg += "\tMove forward   (along x-axis): L\n"
        msg += "\tMove backward  (along x-axis): H\n"
        msg += "\tMove right     (along y-axis): K\n"
        msg += "\tMove left      (along y-axis): J\n"
        msg += "\tYaw positively (along z-axis): U\n"
        msg += "\tYaw negatively (along z-axis): I"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.KeyboardInput>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    """
    Internal helpers.
    """
    def _increase_x_vel(self):
        command = self.env.command_manager.get_command("base_velocity")
        command[:,0] += self.v_x_sensitivity
        print(f"""Base velocity command: {self.env.command_manager.get_command("base_velocity")}""")

    def _decrease_x_vel(self):
        command = self.env.command_manager.get_command("base_velocity")
        command[:,0] -= self.v_x_sensitivity
        print(f"""Base velocity command: {self.env.command_manager.get_command("base_velocity")}""")

    def _increase_y_vel(self):
        command = self.env.command_manager.get_command("base_velocity")
        command[:,1] += self.v_y_sensitivity
        print(f"""Base velocity command: {self.env.command_manager.get_command("base_velocity")}""")

    def _decrease_y_vel(self):
        command = self.env.command_manager.get_command("base_velocity")
        command[:,1] -= self.v_y_sensitivity
        print(f"""Base velocity command: {self.env.command_manager.get_command("base_velocity")}""")

    def _increase_yaw_vel(self):
        command = self.env.command_manager.get_command("base_velocity")
        command[:,2] += self.yaw_sensitivity
        print(f"""Base velocity command: {self.env.command_manager.get_command("base_velocity")}""")

    def _decrease_yaw_vel(self):
        command = self.env.command_manager.get_command("base_velocity")
        command[:,2] -= self.yaw_sensitivity
        print(f"""Base velocity command: {self.env.command_manager.get_command("base_velocity")}""")
    
    def _increase_step_period(self):
        step_period = self.env.command_manager.get_command("step_command")
        step_period += 1
        print(f"""Step period: {step_period}""")

    def _decrease_step_period(self):
        step_period = self.env.command_manager.get_command("step_command")
        step_period -= 1
        print(f"""Step period: {step_period}""")

    def _shoot(self):
        brl_mdp.shoot_projectile(self.env)

    def _increase_force_disturbance(self):
        self.disturbance += 0.5
        print(f"Increase disturbance to {self.disturbance}")

    def _decrease_force_disturbance(self):
        self.disturbance -= 0.5
        print(f"Decrease disturbance to {self.disturbance}")

    def _force_disturbance(self):
        # brl_mdp.apply_external_force_torque_disturbance(self.env, None, force_range=(-15.0, 15.0), torque_range=(-1.5, 1.5))
        # brl_mdp.apply_external_force_torque_disturbance(self.env, None, force_range=(30.0, 30.0), torque_range=(0., 0.))
        # brl_mdp.apply_external_force_torque_disturbance(self.env, None, force_range=(0.0, 0.0), torque_range=(-3.0, -3.0))
        brl_mdp.apply_external_force_torque_disturbance(self.env, None, force_range=(0.0, 0.0), torque_range=(self.disturbance, self.disturbance))
        print("\033[94mApply external force and torque!\033[0m")

    def _get_key_action_map(self):
        """ Returns the key action map for the keyboard.

        Returns:
            dict: The key action map.
        """
        key_action_map = {
            "R": self._reset,
            "X": self._screenshot,
            "W": self._record,
            "C": self._shoot,
            "F": self._force_disturbance,
            "L": self._increase_x_vel,
            "H": self._decrease_x_vel,
            "K": self._increase_y_vel,
            "J": self._decrease_y_vel,
            "U": self._increase_yaw_vel,
            "I": self._decrease_yaw_vel,
            "M": self._increase_step_period,
            "N": self._decrease_step_period,
            "SHIFT+M": self._increase_force_disturbance,
            "SHIFT+N": self._decrease_force_disturbance,
        }
        return key_action_map

class MPCKeyboard(VanillaKeyboard):
    r"""A MPC keyboard controller for basic functionality.

    This class is designed to provide a keyboard controller for mobile base (such as quadrupeds) and mpc controller.
    It uses the Omniverse keyboard interface to listen to keyboard events.

    Key bindings:
        ====================== ================
        Basic Command          Key 
        ====================== ================
        Reset                  R       
        Screenshot             X
        Record                 W
        ====================== ================ ================
        Velocity Command       Key (+ve axis)   Key (-ve axis)
        ====================== ================ ================
        Move along x-axis      L                H
        Move along y-axis      K                J
        Rotate along z-axis    U                I
        ====================== ================ ================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, env, controller, v_x_sensitivity: float = 0.5, v_y_sensitivity: float = 0.5, yaw_sensitivity: float = torch.pi/6):
        """Initialize the keyboard layer.

        Args:
            env: The environment object.
            controller: The mpc controller object.
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 0.2.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 0.2.
            yaw_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to pi/6.
        """
        super().__init__(env, v_x_sensitivity, v_y_sensitivity, yaw_sensitivity)
        self.controller = controller

    """
    Internal helpers.
    """
    def _reset(self):
        self.env.reset()
        self.controller.reset()
        print("Environment and Controller reset.")