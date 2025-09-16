# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
import math

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def throw_ball_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # get robot root state
    robot_root_pos = env.robot.data.root_pos_w[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = robot_root_pos + rand_samples[:, 0:3]
    
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def shoot_projectile(
    env: ManagerBasedEnv,
):
    """Shoot a projectile from a random position around a sphere (radius 1m) towards the robot's center."""
    
    try:
        asset: RigidObject = env.scene['ball']
        env_ids = torch.arange(env.num_envs, device=env.device)

        # Get the robot's root position in world coordinates
        robot_root_pos = env.robot.data.root_pos_w[env_ids].clone()  # [num_envs, 3]

        # Generate random points on a sphere of radius 1m around the robot
        theta = torch.rand(env.num_envs, device=env.device) * 2 * torch.pi  # Azimuthal angle (0 to 2π)
        phi = torch.acos(torch.rand(env.num_envs, device=env.device))  # Polar angle (0 to π/2)

        # Convert spherical to Cartesian coordinates (random points on the sphere)
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)

        # Scale to a sphere of radius 1m and add to the robot's position
        spawn_positions = robot_root_pos + torch.stack([x, y, z], dim=-1)

        # Compute velocity direction (toward the robot's center)
        velocity_directions = robot_root_pos - spawn_positions  # Vector pointing to the robot
        velocity_directions = torch.nn.functional.normalize(velocity_directions, dim=-1)  # Normalize

        # Assign velocity magnitude randomly between 1 and 2 m/s
        velocity_magnitudes = torch.rand(env.num_envs, device=env.device) + 3  # Uniformly between [1, 2]
        projectile_velocities = velocity_directions * velocity_magnitudes.unsqueeze(-1)

        # Assign zero orientation and angular velocity
        orientations = math_utils.quat_from_euler_xyz(torch.zeros(env.num_envs, device=env.device),
                                                        torch.zeros(env.num_envs, device=env.device),
                                                        torch.zeros(env.num_envs, device=env.device))
        angular_velocities = torch.zeros_like(projectile_velocities, device=env.device)

        # Set projectile positions and velocities in the simulation
        pose = torch.hstack((spawn_positions, orientations))
        velocities = torch.hstack((projectile_velocities, angular_velocities))

        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        
        print("\033[94mShooting the ball!\033[0m")

    except KeyError:
        # If 'ball' is not in the scene, print an error message
        print("\033[91mError: 'ball' not found in the scene.\033[0m")

def apply_external_force_torque_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):
    """Apply the random external force and torque to the robot once per call.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    body_ids, body_names = env.robot.find_bodies(asset_cfg.body_names)

    # resolve number of bodies
    num_bodies = len(body_ids) if isinstance(body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device) / env.physics_dt
    torques = math_utils.sample_uniform(*torque_range, size, asset.device) / env.physics_dt

    _external_force_b = torch.zeros((env.scene.num_envs, asset.num_bodies, 3), device=env.device)
    _external_torque_b = torch.zeros_like(_external_force_b)

    indices = torch.tensor(body_ids, dtype=torch.long, device=env.device).repeat(len(env_ids), 1) + \
              env_ids.unsqueeze(1) * asset.num_bodies
    indices = indices.view(-1)

    _external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
    _external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)

    asset.root_physx_view.apply_forces_and_torques_at_position(
        force_data=_external_force_b.view(-1, 3),
        torque_data=_external_torque_b.view(-1, 3),
        position_data=None,
        indices=env_ids,
        is_global=False,
    )

def make_force_grid(
    force_range: tuple[float, float],
    num_envs: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Create an evenly-spaced 2-D grid of forces over [lower, upper] for fx and fy.

    Args:
        force_range: (lower, upper) bounds for each axis.
        num_envs:    Total number of envs. Must be a perfect square.
        device:      CUDA / CPU device for the output tensor.

    Returns
        forces: Tensor of shape (num_envs, 1, 3) with rows (fx, fy, 0).
    """
    lower, upper = force_range
    k = int(math.sqrt(num_envs))
    if k * k != num_envs:
        raise ValueError("num_envs must be a perfect square to form a √nx√n grid.")

    # 1-D linspace for each axis
    axis_vals = torch.linspace(lower, upper, k, device=device)

    # Cartesian product → k×k grid, then flatten to (num_envs,)
    fx, fy = torch.meshgrid(axis_vals, axis_vals, indexing="ij")
    fx = fx.reshape(-1)
    fy = fy.reshape(-1)

    # Stack and add the zero-torque dimension → (n, 3), then unsqueeze → (n,1,3)
    forces = torch.stack([fx, fy, torch.zeros_like(fx)], dim=1).unsqueeze(1)
    return forces

def apply_grid_force_torque_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):
    """Apply the grid external force and torque to the robot once per call.
    
    The grid is defined by the number of rows and columns, and the forces are applied in a grid pattern.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    body_ids, body_names = env.robot.find_bodies(asset_cfg.body_names)

    # generate a grid of forces and torques
    forces = make_force_grid(force_range, env.scene.num_envs, asset.device) / env.physics_dt
    torques = make_force_grid(torque_range, env.scene.num_envs, asset.device) / env.physics_dt

    _external_force_b = torch.zeros((env.scene.num_envs, asset.num_bodies, 3), device=env.device)
    _external_torque_b = torch.zeros_like(_external_force_b)

    indices = torch.tensor(body_ids, dtype=torch.long, device=env.device).repeat(len(env_ids), 1) + \
              env_ids.unsqueeze(1) * asset.num_bodies
    indices = indices.view(-1)

    _external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
    _external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)

    asset.root_physx_view.apply_forces_and_torques_at_position(
        force_data=_external_force_b.view(-1, 3),
        torque_data=_external_torque_b.view(-1, 3),
        position_data=None,
        indices=env_ids,
        is_global=False,
    )