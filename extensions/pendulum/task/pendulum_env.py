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
from extensions.humanoid.utils import VanillaKeyboard

class PendulumEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _init_buffers(self):
        super()._init_buffers()

    def _post_physics_step_callback(self):
        self.pendulum_is_upright = torch.cos(self.robot.data.joint_pos[:,0]) > 0.9

    def _setup_keyboard_interface(self):
        self.keyboard_interface = VanillaKeyboard(self)


