# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import traceback

import carb
from rsl_rl.runners import ModularOnPolicyRunner

# Import extensions to set up environment tasks
import extensions.humanoid  # noqa: F401
import isaaclab_tasks  # noqa: F401
from extensions import ISAACLAB_BRL_ROOT_DIR

from extensions.humanoid.utils import ContactForcesLivePlotter, ScreenShotter

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg, set_registry_to_original_files
from isaaclab_rl.rsl_rl import (
    RslRlModularOnPolicyRunnerCfg,
    RslRlModularVecEnvWrapper,
)


def main():
    """Play with RSL-RL agent."""
    if args_cli.load_files:
        set_registry_to_original_files(args_cli.task, args_cli.load_run)
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env_cfg.seed = agent_cfg.seed
    env_cfg.record = RECORD
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlModularVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # load previously trained model
    ppo_runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    leg_policy, arm_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # * export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(ISAACLAB_BRL_ROOT_DIR, 'logs', 'rsl_rl', agent_cfg.experiment_name, 'exported')
        token = resume_path.split('/')
        model_name = f'{token[-2]}_{token[-1].split(".")[0]}'
        ppo_runner.export(path, model_name)
        print(f'Exported policy model "{model_name}" to: ', path)

    # initialize custom functionality for logging
    log_states = ["root_lin_vel_w"] # "step_length", "step_width", "base_lin_vel", "base_pos", "dof_pos", 
    log_commands = ["vel_command"]
    robot_index = 0  # which robot is used for logging

    screenshotter = ScreenShotter(env.unwrapped, resume_path)
    if LIVE_PLOT:
        contactForcesLivePlotter = ContactForcesLivePlotter(env.unwrapped)

    # reset environment
    obs_dict = env.get_observations()
    leg_actor_obs, arm_actor_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            leg_actions = leg_policy(leg_actor_obs)
            arm_actions = arm_policy(arm_actor_obs)
            actions = torch.cat((leg_actions, arm_actions), dim=1)

            # env stepping
            obs_dict, _, _, _, _, _ = env.step(actions)
            leg_actor_obs, arm_actor_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]

        if env.unwrapped.screenshot:
            image = env.viewport_camera_image
            screenshotter.screenshot(image)
            env.unwrapped.screenshot = False

        if CUSTOM_COMMANDS:
            if env.unwrapped.common_step_counter == 1:
                env.unwrapped.vel_command[:, 0] = 1.0
                env.unwrapped.vel_command[:, 1] = 0.
                env.unwrapped.vel_command[:, 2] = 0.
            elif env.unwrapped.common_step_counter == 200:
                env.unwrapped.vel_command[:, 0] = 1.0
                env.unwrapped.vel_command[:, 1] = 0.
                env.unwrapped.vel_command[:, 2] = 1.0
            elif env.unwrapped.common_step_counter == 400:
                env.unwrapped.vel_command[:, 0] = 0.
                env.unwrapped.vel_command[:, 1] = 0.
                env.unwrapped.vel_command[:, 2] = 1.0
            elif env.unwrapped.common_step_counter == 600:
                env.unwrapped.record_done = True

        if LIVE_PLOT:
            # custom plotting
            contactForcesLivePlotter.log(env.unwrapped.contact_forces[robot_index])
            contactForcesLivePlotter.plot()

        if RECORD:
            if env.unwrapped.record_done:
                env.unwrapped.recorder.save(resume_path)
                simulation_app.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True # True, False
    CUSTOM_COMMANDS = False # True, False
    LIVE_PLOT = False # True, False
    RECORD = True # True, False

    # args_cli.load_files = True # True, False
    # args_cli.load_run = '2025-05-23_00-34-58'
    # args_cli.checkpoint = 'model_1000.pt'
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
