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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import math
import matplotlib.pyplot as plt
from pathlib import Path

import carb
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners import ModularOnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from extensions import ISAACLAB_BRL_ROOT_DIR
import extensions.humanoid.mdp as brl_mdp

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg, set_registry_to_original_files
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper,
    RslRlModularOnPolicyRunnerCfg, RslRlModularVecEnvWrapper
)


def evaluate_one(task: str,
                 run: str,
                 force_range: torch.Tensor,
                 torque_range: torch.Tensor,
                 device: str = "cuda:0") -> torch.Tensor:
    """Run one 5-s rollout, return Bool grid (num_envs,) indicating survival."""
    if args_cli.load_files:
        set_registry_to_original_files(task, args_cli.load_run)
    omni.usd.get_context().new_stage()

    if "modular" in task:
        # -----------------  make env & runner  ----------------------------------
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task, device=device, num_envs=NUM_ENVS)
        agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task, args_cli)

        env_cfg.terminations.leg.base_termination = None
        env_cfg.seed = agent_cfg.seed
        env_cfg.record = RECORD
        env = gym.make(task, cfg=env_cfg)
        env = RslRlModularVecEnvWrapper(env)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, run, agent_cfg.load_checkpoint)

        ppo_runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        ppo_runner.load(resume_path)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        leg_policy, arm_policy = ppo_runner.get_inference_policy(device=device)

    else:
        # -----------------  make env & runner  ----------------------------------
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task, device=device, num_envs=NUM_ENVS)
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task, args_cli)

        env_cfg.terminations.leg.base_termination = None
        env_cfg.seed = agent_cfg.seed
        env_cfg.record = RECORD
        env = gym.make(task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, run, agent_cfg.load_checkpoint)

        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        ppo_runner.load(resume_path)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=device)

    obs_dict = env.get_observations()
    survived = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for step in range(SIM_STEPS):
            # ---- choose actions -------------------------------------------
            if "modular" in task:
                leg_actor_obs, arm_actor_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]
                leg_actions = leg_policy(leg_actor_obs)
                arm_actions = arm_policy(arm_actor_obs)
                actions = torch.cat((leg_actions, arm_actions), dim=1)
            else:
                actor_obs = obs_dict["actor"]
                actions = policy(actor_obs)

            # ---- step env --------------------------------------------------
            obs_dict, _, done, *_ = env.step(actions)
            survived &= ~done.bool()

            # ---- apply single push at 1 s ---------------------------------
            if step == DISTURB_STEP:
                brl_mdp.apply_grid_force_torque_disturbance(
                    env.unwrapped, None,
                    force_range  = force_range,
                    torque_range = torque_range
                )
            # ---- record ---------------------------------------------------
            if RECORD and step == SIM_STEPS - 1:
                env.unwrapped.recorder.save(resume_path)

    env.close()
    ppo_runner.close()
    return survived                    # shape (num_envs,)

def evaluate_cycles(task: str,
                    run: str,
                    force_range: tuple[float, float],
                    torque_range: tuple[float, float],
                    cycles: int = 10,
                    device: str = "cuda:0") -> torch.Tensor:
    """
    Run *cycles* x 5 s with one checkpoint.
    Return Bool sum (NUM_ENVS,) indicating how many cycles each env survived.
    """
    # ------------------------------------------------------------------ #
    if args_cli.load_files:
        set_registry_to_original_files(task, run)
    omni.usd.get_context().new_stage()

    # ---- build env & policy (identical to your old branches) ----------
    if "modular" in task:
        # -----------------  make env & runner  ----------------------------------
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task, device=device, num_envs=NUM_ENVS)
        agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task, args_cli)

        env_cfg.terminations.leg.base_termination = None
        env_cfg.observations.leg_actor.enable_corruption = True # True, False
        env_cfg.observations.leg_actor.enable_corruption = True # True, False
        env_cfg.seed = agent_cfg.seed
        env_cfg.record = RECORD
        env = gym.make(task, cfg=env_cfg)
        env = RslRlModularVecEnvWrapper(env)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, run, agent_cfg.load_checkpoint)

        ppo_runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        ppo_runner.load(resume_path)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        leg_policy, arm_policy = ppo_runner.get_inference_policy(device=device)

        def act(obs):
            return torch.cat((leg_policy(obs["leg_actor"]), arm_policy(obs["arm_actor"])), dim=1)
    else:
        # -----------------  make env & runner  ----------------------------------
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task, device=device, num_envs=NUM_ENVS)
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task, args_cli)

        env_cfg.terminations.leg.base_termination = None
        env_cfg.observations.actor.enable_corruption = True # True, False
        env_cfg.seed = agent_cfg.seed
        env_cfg.record = RECORD
        env = gym.make(task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, run, agent_cfg.load_checkpoint)

        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        ppo_runner.load(resume_path)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=device)

        def act(obs):
            return policy(obs["actor"])

    # ------------------------------------------------------------------ #
    survived_counter = torch.zeros(NUM_ENVS, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for cycle in range(cycles):
            obs = env.reset()                           # manual hard reset
            survived = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)
            for step in range(SIM_STEPS):               # 5 s segment
                actions = act(obs)
                obs, _, done, *_ = env.step(actions)
                survived &= ~done.bool()

                if step == DISTURB_STEP:                # 1 s mark
                    brl_mdp.apply_grid_force_torque_disturbance(
                        env.unwrapped, None,
                        force_range=force_range,
                        torque_range=torque_range
                    )

            # ---- record ---------------------------------------------------
            if RECORD and cycle == cycles - 1:
                env.unwrapped.recorder.save(resume_path)

            survived_counter += survived.float()        # accumulate success

    env.close()
    ppo_runner.close()
    return survived_counter                         # (NUM_ENVS,)

def aggregate(task: str, run: str, *, force_on: bool) -> torch.Tensor:
    """Return (side,side) probability map after running all runs."""
    side    = int(math.sqrt(NUM_ENVS))
    counts  = torch.zeros(NUM_ENVS, dtype=torch.float32, device=args_cli.device)

    force_range  = FORCE_RANGE if force_on else (0.0, 0.0)
    torque_range = TORQUE_RANGE if not force_on else (0.0, 0.0)

    counts = evaluate_cycles(task, run,
                             force_range=force_range,
                             torque_range=torque_range,
                             cycles=NUM_CYCLES,
                             device=args_cli.device)

    probs = (counts / NUM_CYCLES).reshape(side, side)
    return probs

def plot_heatmaps(task: str,
                  p_force: torch.Tensor,
                  p_torque: torch.Tensor):
    """Draw one 1x2 panel:   [force | torque]   and save as PDF."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
    panels = [
        ("Force",   p_force,  FORCE_RANGE,  "Greens", "Fx [N]", "Fy [N]"),
        ("Torque",  p_torque, TORQUE_RANGE, "Blues",  "τx [Nm]", "τy [Nm]"),
    ]

    for ax, (title, grid, rng, cmap, xlabel, ylabel) in zip(axes, panels):
        im = ax.imshow(
            grid.cpu().numpy(),
            origin="lower",
            extent=[rng[0], rng[1], rng[0], rng[1]],
            cmap=cmap, vmin=0, vmax=1
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Survival rate")

    fig.suptitle(f"{task} - push-recovery survival rate")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = SAVE_FIG_PATH / f"{task}_force_torque_survival.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[✓] Saved {out}")

def save_probs(vanilla_p_force: torch.Tensor, vanilla_p_torque: torch.Tensor,
               modular_p_force: torch.Tensor, modular_p_torque: torch.Tensor):
    """Save the probability maps to disk."""
    data = {
        "vanilla_force":  vanilla_p_force.cpu().numpy(),
        "vanilla_torque": vanilla_p_torque.cpu().numpy(),
        "modular_force":  modular_p_force.cpu().numpy(),
        "modular_torque": modular_p_torque.cpu().numpy(),
    }
    out_path = SAVE_DATA_PATH / "humanoid_survival_probs.pkl"
    dump_pickle(out_path, data)
    print(f"[✓] Saved survival probabilities to {out_path}")


if __name__ == "__main__":
    SAVE_FIG_PATH  = Path(ISAACLAB_BRL_ROOT_DIR) / "scripts" / "plotting" / "figures"
    SAVE_DATA_PATH = Path(ISAACLAB_BRL_ROOT_DIR) / "scripts" / "plotting" / "data"

    NUM_ENVS     = args_cli.num_envs
    FORCE_RANGE  = (-70.0, 70.0)  # Fx / Fy (N)
    TORQUE_RANGE = (-15.0, 15.0)  # τx / τy (Nm)
    SIM_STEPS    = 250  # 5 seconds
    DISTURB_STEP = 50 # 1 seconds
    NUM_CYCLES   = 10 # number of cycles to run
    RECORD       = False # True, False

    args_cli.load_files = True

    # vanilla_runs = ['2025-06-06_00-36-58', '2025-06-06_00-56-43', '2025-06-06_01-16-33', '2025-06-06_01-36-21']
    # modular_runs = ['2025-06-05_17-06-40', '2025-06-05_17-29-43', '2025-06-05_17-52-58', '2025-06-05_18-16-12']
    vanilla_runs = '2025-06-09_16-48-25'
    modular_runs = '2025-06-10_02-44-32'

    # -------- vanilla -------------------------------------------------------
    vanilla_p_force  = aggregate("humanoid_vanilla_play", vanilla_runs, force_on=True)
    vanilla_p_torque = aggregate("humanoid_vanilla_play", vanilla_runs, force_on=False)
    plot_heatmaps("vanilla", vanilla_p_force, vanilla_p_torque)

    # -------- modular -------------------------------------------------------
    modualr_p_force  = aggregate("humanoid_full_modular_play", modular_runs, force_on=True)
    modualr_p_torque = aggregate("humanoid_full_modular_play", modular_runs, force_on=False)
    plot_heatmaps("modular", modualr_p_force, modualr_p_torque)

    save_probs(vanilla_p_force, vanilla_p_torque, modualr_p_force, modualr_p_torque)

    simulation_app.close()
