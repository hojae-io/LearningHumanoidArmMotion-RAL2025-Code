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

class MOD:
    SHIFT = 1
    CTRL  = 2
    ALT   = 4

class BaseKeyboard:
    r"""A base keyboard controller for basic functionality.

    This class is designed to provide a keyboard controller for mobile base (such as quadrupeds).
    It uses the Omniverse keyboard interface to listen to keyboard events.

    Key bindings:
        ====================== ================
        Basic Command          Key 
        ====================== ================
        Reset                  R       
        Screenshot             X
        Record                 W
        ====================== ================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, env):
        """Initialize the keyboard layer.

        Args:
            env: The environment object.
        """
        self.env = env
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"\tKeyboard Controller: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset : R\n"
        msg += "\tScreenshot : X\n"
        msg += "\tRecord : W\n"
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

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.
        """
        pass

    """
    Internal helpers.
    """
    def _reset(self):
        self.env.reset()
        print("Environment reset.")

    def _screenshot(self):
        self.env.screenshot = True
        print("Screenshot taken.")

    def _record(self):
        self.env.record_done = True
        print("Recording.")

    def _get_key_action_map(self):
        """Returns the key action map for the keyboard.

        Returns:
            dict: The key action map.
        """
        key_action_map = {
            "R": self._reset,
            "X": self._screenshot,
            "W": self._record,
        }
        return key_action_map
    
    def _event_to_token(self, event) -> str:
        mods = []
        if event.modifiers & MOD.SHIFT:
            mods.append("SHIFT")
        if event.modifiers & MOD.CTRL:
            mods.append("CTRL")
        if event.modifiers & MOD.ALT:
            mods.append("ALT")

        key_name = event.input.name              # "K", "F", …
        return "+".join(mods + [key_name]) if mods else key_name

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_action_map = self._get_key_action_map()
            token = self._event_to_token(event)                 # "K", "Shift+K", …
            action = key_action_map.get(token)
            if action:
                action()
        
        return True
